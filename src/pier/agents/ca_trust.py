"""Custom CA trust for installed agents running inside the sandbox.

When the inference gateway is served with a private certificate authority, every
agent runtime inside the sandbox has to trust that CA -- and each runtime looks
in a different place. Point ``PIER_EXTRA_CA_CERTS`` at a PEM bundle on the host
and Pier appends an install step that registers it with the sandbox's OS trust
store, then exports the per-runtime CA variables for every installed agent.

Two files are written into the sandbox:

``/usr/local/share/pier/extra-ca.pem``
    Only the extra certificates. This is for consumers that *add* to their
    default trust store (``NODE_EXTRA_CA_CERTS``, ``CODEX_CA_CERTIFICATE``).

``/usr/local/share/pier/ca-bundle.pem``
    The distro roots plus the extra certificates. This is for consumers that
    *replace* their trust store with the file they are given
    (``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``).

Pier's filtered-egress proxy tunnels HTTPS with ``CONNECT`` and never re-signs
the origin certificate, so the agent validates the gateway's real certificate
regardless of network mode and needs this either way.
"""

from __future__ import annotations

import base64
import os
import ssl
from pathlib import Path

from pier.models.agent.install import AgentInstallSpec, InstallStep

EXTRA_CA_CERTS_ENV = "PIER_EXTRA_CA_CERTS"

CA_DIR = "/usr/local/share/pier"
EXTRA_CA_PATH = f"{CA_DIR}/extra-ca.pem"
CA_BUNDLE_PATH = f"{CA_DIR}/ca-bundle.pem"

# Probed in order when assembling ``ca-bundle.pem``.
SYSTEM_CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian, Ubuntu, Alpine
    "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL, Fedora, Amazon Linux
    "/etc/ssl/cert.pem",  # assorted musl images
)

_BEGIN_CERTIFICATE = "-----BEGIN CERTIFICATE-----"
_END_CERTIFICATE = "-----END CERTIFICATE-----"


class ExtraCaCertsError(RuntimeError):
    """Raised when ``PIER_EXTRA_CA_CERTS`` is set but unusable."""


def extra_ca_certs_configured() -> bool:
    """Whether ``PIER_EXTRA_CA_CERTS`` names a path.

    Deliberately cheap: this is called on every agent command and only checks
    the variable. The file itself is read and validated once, when the install
    spec is built, so a bad path fails before the trial starts.
    """
    return bool(os.environ.get(EXTRA_CA_CERTS_ENV, "").strip())


def read_extra_ca_certs() -> str | None:
    """Return the configured PEM bundle, or ``None`` when not configured."""
    raw = os.environ.get(EXTRA_CA_CERTS_ENV, "").strip()
    if not raw:
        return None

    path = Path(raw).expanduser()
    try:
        pem = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExtraCaCertsError(
            f"{EXTRA_CA_CERTS_ENV} points at {path}, which could not be read: {exc}"
        ) from exc

    certificates = split_pem_certificates(pem)
    if not certificates:
        raise ExtraCaCertsError(
            f"{EXTRA_CA_CERTS_ENV} points at {path}, which contains no "
            f"'{_BEGIN_CERTIFICATE}' block. It must be a PEM file holding one or "
            "more certificates."
        )
    _reject_unparseable_certificates(certificates, path)
    return pem


def _reject_unparseable_certificates(certificates: list[str], path: Path) -> None:
    """Fail on PEM blocks that OpenSSL cannot parse as certificates.

    Matching delimiters are not enough: a block with a corrupt body would be
    embedded in the install step, silently skipped by ``update-ca-certificates``
    (which is best effort), and then appended to the merged bundle -- where it can
    break parsing of the whole file and take *all* TLS down with it. Validating
    here, on the host, turns that into an error before the trial starts. The check
    runs through the same OpenSSL parser that will read the bundle in the sandbox.
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    for index, certificate in enumerate(certificates, start=1):
        try:
            context.load_verify_locations(cadata=certificate)
        except (ssl.SSLError, ValueError) as exc:
            raise ExtraCaCertsError(
                f"{EXTRA_CA_CERTS_ENV} points at {path}, whose certificate "
                f"#{index} could not be parsed: {exc}. Every "
                f"'{_BEGIN_CERTIFICATE}' block must hold a valid certificate."
            ) from exc


def split_pem_certificates(pem: str) -> list[str]:
    """Split a PEM bundle into its individual certificate blocks.

    Anything outside a ``CERTIFICATE`` block is dropped, so bundles that also
    carry CRLs, private keys, or OpenSSL's ``TRUSTED CERTIFICATE`` sections are
    accepted. Debian's ``update-ca-certificates`` reads only the first
    certificate of a ``.crt`` file, which is why each block gets its own file.
    """
    certificates: list[str] = []
    current: list[str] | None = None
    for line in pem.splitlines():
        stripped = line.strip()
        if stripped == _BEGIN_CERTIFICATE:
            current = [stripped]
        elif stripped == _END_CERTIFICATE and current is not None:
            current.append(stripped)
            certificates.append("\n".join(current) + "\n")
            current = None
        elif current is not None:
            current.append(stripped)
    return certificates


def _b64(text: str) -> str:
    """Encode ``text`` for embedding in a single-quoted shell string."""
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def ca_trust_install_script(pem: str) -> str:
    """Shell script that installs ``pem`` into the sandbox's trust store."""
    certificates = split_pem_certificates(pem)
    anchor_writes = "\n".join(
        f"  printf '%s' '{_b64(certificate)}' | base64 -d "
        f'> "$anchor_dir/pier-extra-ca-{index}.crt"'
        for index, certificate in enumerate(certificates, start=1)
    )
    probe = " ".join(f'"{bundle}"' for bundle in SYSTEM_CA_BUNDLES)

    return f"""set -eu
mkdir -p {CA_DIR}
printf '%s' '{_b64("".join(certificates))}' | base64 -d > {EXTRA_CA_PATH}

# Register with the OS trust store so opaque native binaries pick the CA up
# too. Best effort by design: the environment variables Pier exports are what
# the agent runtimes actually read, and they only need the files above.
if command -v update-ca-certificates >/dev/null 2>&1; then
  anchor_dir=/usr/local/share/ca-certificates
  mkdir -p "$anchor_dir"
  # Clear anchors from an earlier run so a shrinking bundle does not leave a
  # dropped root behind in a reused image.
  rm -f "$anchor_dir"/pier-extra-ca-*.crt
{anchor_writes}
  update-ca-certificates \
    || echo "pier: warning: update-ca-certificates failed; the OS trust store" \
      "may not contain PIER_EXTRA_CA_CERTS. The exported CA variables still" \
      "point at the files above." >&2
elif command -v update-ca-trust >/dev/null 2>&1; then
  anchor_dir=/etc/pki/ca-trust/source/anchors
  mkdir -p "$anchor_dir"
  # Clear anchors from an earlier run so a shrinking bundle does not leave a
  # dropped root behind in a reused image.
  rm -f "$anchor_dir"/pier-extra-ca-*.crt
{anchor_writes}
  update-ca-trust extract \
    || echo "pier: warning: update-ca-trust failed; the OS trust store may not" \
      "contain PIER_EXTRA_CA_CERTS. The exported CA variables still point at" \
      "the files above." >&2
fi

# Assemble the replace-style bundle: distro roots first, then the extras. If the
# trust-store update above worked the extras appear twice, which is harmless.
: > {CA_BUNDLE_PATH}
for bundle in {probe}; do
  if [ -s "$bundle" ]; then
    cat "$bundle" >> {CA_BUNDLE_PATH}
    break
  fi
done

# An image with no distro roots would otherwise leave SSL_CERT_FILE and
# REQUESTS_CA_BUNDLE pointing at a bundle holding only the extras, which would
# strip certifi's roots out from under the Python agents. Fall back to certifi so
# the bundle never has *less* public trust than the runtime started with.
if [ ! -s {CA_BUNDLE_PATH} ] && command -v python3 >/dev/null 2>&1; then
  python3 -c 'import certifi, sys; sys.stdout.write(open(certifi.where()).read())' \
    >> {CA_BUNDLE_PATH} 2>/dev/null || true
fi

# Nothing public to merge: the image has no trust store and no certifi. The
# extras below still work for the gateway, but anything reading the bundle loses
# public trust, so say so in the build log rather than failing mysteriously later.
if [ ! -s {CA_BUNDLE_PATH} ]; then
  echo "pier: warning: no public CA roots found in this image; {CA_BUNDLE_PATH}" \
    "will contain only PIER_EXTRA_CA_CERTS. Install ca-certificates in the task" \
    "image if the agent needs public HTTPS." >&2
fi

cat {EXTRA_CA_PATH} >> {CA_BUNDLE_PATH}
chmod 0644 {EXTRA_CA_PATH} {CA_BUNDLE_PATH}
"""


def ca_trust_install_step(pem: str) -> InstallStep:
    """Wrap :func:`ca_trust_install_script` as a root install step."""
    return InstallStep(user="root", run=ca_trust_install_script(pem))


def with_extra_ca_certs(spec: AgentInstallSpec) -> AgentInstallSpec:
    """Append the CA trust step to ``spec`` when one is configured.

    The step runs last so that it can rely on ``ca-certificates`` and the
    distro's trust tooling, which the agent's own steps install. The agent's
    downloads target public hosts and need the public roots, not this CA.

    The returned spec has a different :meth:`AgentInstallSpec.fingerprint`, so
    changing the CA correctly invalidates cached sandbox images.
    """
    pem = read_extra_ca_certs()
    if pem is None:
        return spec
    return spec.model_copy(update={"steps": [*spec.steps, ca_trust_install_step(pem)]})


def ca_trust_env() -> dict[str, str]:
    """CA variables for agent commands, empty when no CA is configured.

    ``*_CA_BUNDLE`` and ``SSL_CERT_FILE`` replace the default trust store, so
    they get the merged bundle. ``NODE_EXTRA_CA_CERTS`` and
    ``CODEX_CA_CERTIFICATE`` add to it, so they get the extras alone.
    """
    if not extra_ca_certs_configured():
        return {}
    return {
        "SSL_CERT_FILE": CA_BUNDLE_PATH,
        "REQUESTS_CA_BUNDLE": CA_BUNDLE_PATH,
        "CURL_CA_BUNDLE": CA_BUNDLE_PATH,
        "GIT_SSL_CAINFO": CA_BUNDLE_PATH,
        "NODE_EXTRA_CA_CERTS": EXTRA_CA_PATH,
        "CODEX_CA_CERTIFICATE": EXTRA_CA_PATH,
    }
