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
    The union of the public trust sources found in the image -- every readable
    distro bundle, plus certifi for each interpreter in :data:`CERTIFI_PYTHONS`
    -- followed by the extra certificates. This is for consumers that *replace*
    their trust store with the file they are given (``SSL_CERT_FILE``,
    ``REQUESTS_CA_BUNDLE``, ``CURL_CA_BUNDLE``). It is a union rather than the
    first source found because a Python runtime that normally trusts a newer
    certifi must not lose roots merely because the image also ships a distro
    bundle.

    The interpreter list is a bounded guess, so this is not a general guarantee:
    a Python runtime living somewhere unprobed, holding certifi roots no probed
    interpreter has, could still see its trust narrowed by
    ``REQUESTS_CA_BUNDLE``. Closing that properly means either runtime-aware
    trust construction or appending to each discovered certifi in place instead
    of replacing trust through environment variables.

Pier's filtered-egress proxy tunnels HTTPS with ``CONNECT`` and never re-signs
the origin certificate, so the agent validates the gateway's real certificate
regardless of network mode and needs this either way.
"""

from __future__ import annotations

import base64
import hashlib
import os
import ssl
from pathlib import Path
from typing import Any

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

# Interpreters probed for a certifi bundle. Pier's Python agents do not run under
# the system python3 -- antigravity gets its own venv and mini-swe a ``uv tool``
# environment -- and each can carry its own certifi. Unmatched globs stay literal
# and are filtered out by the executable check in the generated script.
#
# The uv paths are globbed across home directories rather than taken from $HOME:
# this step runs as root, but mini-swe installs under the *agent* user, and
# ``[agent].user`` need not be root. Probing only $HOME would silently cover just
# the root case. This is still a bounded guess at where an interpreter lives --
# see the module docstring for the limits of that.
CERTIFI_PYTHONS = (
    "python3",
    "python",
    "/installed-agent/venv/bin/python",
    "/installed-agent/venv/bin/python3",
    "/root/.local/share/uv/tools/*/bin/python3",
    "/home/*/.local/share/uv/tools/*/bin/python3",
    '"$HOME"/.local/share/uv/tools/*/bin/python3',
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
    # Order matters: a lone unterminated block leaves ``certificates`` empty, and
    # reporting "contains no BEGIN CERTIFICATE block" for a file that visibly has
    # one sends the reader looking in the wrong place.
    _reject_incomplete_blocks(pem, certificates, path)
    if not certificates:
        raise ExtraCaCertsError(
            f"{EXTRA_CA_CERTS_ENV} points at {path}, which contains no "
            f"'{_BEGIN_CERTIFICATE}' block. It must be a PEM file holding one or "
            "more certificates."
        )
    _reject_unparseable_certificates(certificates, path)
    return pem


def _reject_incomplete_blocks(pem: str, certificates: list[str], path: Path) -> None:
    """Fail on certificate blocks that open but never close.

    :func:`split_pem_certificates` drops an unterminated block, so a bundle of
    "good certificate followed by a truncated one" would otherwise be silently
    accepted as just the good one -- installing less trust than the operator
    asked for, with no error. Counting the opening delimiters catches that.

    ``TRUSTED CERTIFICATE`` is rejected rather than relabelled: it carries
    OpenSSL trust settings after the certificate, so it is not interchangeable
    with a plain ``CERTIFICATE`` block.
    """
    lines = [line.strip() for line in pem.splitlines()]
    started = lines.count(_BEGIN_CERTIFICATE)
    if started != len(certificates):
        raise ExtraCaCertsError(
            f"{EXTRA_CA_CERTS_ENV} points at {path}, which has {started} "
            f"'{_BEGIN_CERTIFICATE}' block(s) but only {len(certificates)} "
            f"closed by '{_END_CERTIFICATE}'. A truncated certificate would be "
            "silently skipped, so the bundle is rejected."
        )

    if any(line == "-----BEGIN TRUSTED CERTIFICATE-----" for line in lines):
        raise ExtraCaCertsError(
            f"{EXTRA_CA_CERTS_ENV} points at {path}, which contains a "
            "'-----BEGIN TRUSTED CERTIFICATE-----' block. That form carries "
            "OpenSSL trust settings and is not a plain certificate; convert it "
            "first, e.g. 'openssl x509 -in in.pem -out out.pem'."
        )


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
    carry CRLs or private keys are handled. Debian's ``update-ca-certificates``
    reads only the first certificate of a ``.crt`` file, which is why each block
    gets its own file.

    This is a plain splitter: an unterminated block is dropped and
    ``TRUSTED CERTIFICATE`` is not recognised. :func:`read_extra_ca_certs`
    rejects both rather than letting them pass silently.
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
    # Globs must stay unquoted so the shell can expand them.
    python_probe = " ".join(
        candidate if "*" in candidate else f'"{candidate}"'
        for candidate in CERTIFI_PYTHONS
    )

    return f"""set -eu
mkdir -p {CA_DIR}
printf '%s' '{_b64("".join(certificates))}' | base64 -d > {EXTRA_CA_PATH}

# Register with the OS trust store so opaque native binaries pick the CA up too.
# If a distro updater is present it must succeed: a half-updated store can leave
# a rotated-out root still trusted by curl/git/Python via the system bundle while
# Node and Codex only trust the new one, and silently inconsistent trust is worse
# for a benchmark harness than a failed build.
if command -v update-ca-certificates >/dev/null 2>&1; then
  anchor_dir=/usr/local/share/ca-certificates
  mkdir -p "$anchor_dir"
  # Clear anchors from an earlier run so a shrinking bundle does not leave a
  # dropped root behind in a reused image.
  rm -f "$anchor_dir"/pier-extra-ca-*.crt
{anchor_writes}
  if ! update-ca-certificates; then
    echo "pier: error: update-ca-certificates failed; refusing to continue with" \
      "a partially updated trust store." >&2
    exit 1
  fi
elif command -v update-ca-trust >/dev/null 2>&1; then
  anchor_dir=/etc/pki/ca-trust/source/anchors
  mkdir -p "$anchor_dir"
  # Clear anchors from an earlier run so a shrinking bundle does not leave a
  # dropped root behind in a reused image.
  rm -f "$anchor_dir"/pier-extra-ca-*.crt
{anchor_writes}
  if ! update-ca-trust extract; then
    echo "pier: error: update-ca-trust failed; refusing to continue with a" \
      "partially updated trust store." >&2
    exit 1
  fi
fi

# Assemble the replace-style bundle. These variables *replace* a runtime's trust
# store rather than adding to it, so the bundle must union every public source a
# runtime might otherwise have used, not whichever one is found first. Duplicates
# across sources are harmless -- OpenSSL ignores repeats -- so no path is skipped.
: > {CA_BUNDLE_PATH}
for bundle in {probe}; do
  if [ -s "$bundle" ]; then
    cat "$bundle" >> {CA_BUNDLE_PATH}
  fi
done

# certifi, per interpreter. The Python that will make the request is often not
# the system python3: pier installs antigravity into its own venv and mini-swe
# through `uv tool`, each of which can carry its own certifi. Probing only
# python3 could drop roots those runtimes already trusted. This step runs last in
# the spec, so those interpreters already exist by now.
for py in {python_probe}; do
  if [ -x "$py" ] || command -v "$py" >/dev/null 2>&1; then
    "$py" -c 'import certifi, sys; sys.stdout.write(open(certifi.where()).read())' \
      >> {CA_BUNDLE_PATH} 2>/dev/null || true
  fi
done

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
    changing the CA correctly invalidates cached sandbox images. An explicit
    ``cache_key`` short-circuits that fingerprint, so it is extended with a digest
    of the generated step -- not merely of the PEM -- otherwise a change to
    :func:`ca_trust_install_script` would reuse an image built by the previous
    version of the install logic against the same certificates.
    """
    pem = read_extra_ca_certs()
    if pem is None:
        return spec

    step = ca_trust_install_step(pem)
    update: dict[str, Any] = {"steps": [*spec.steps, step]}
    if spec.cache_key:
        digest = hashlib.sha256(step.run.encode("utf-8")).hexdigest()[:16]
        update["cache_key"] = f"{spec.cache_key}-ca-{digest}"
    return spec.model_copy(update=update)


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
