import base64
import hashlib
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from pier.agents.base import BaseAgent
from pier.agents.ca_trust import (
    CA_BUNDLE_PATH,
    CERTIFI_PYTHONS,
    EXTRA_CA_CERTS_ENV,
    EXTRA_CA_PATH,
    SYSTEM_CA_BUNDLES,
    ExtraCaCertsError,
    ca_trust_env,
    ca_trust_install_script,
    read_extra_ca_certs,
    split_pem_certificates,
    with_extra_ca_certs,
)
from pier.agents.factory import AgentFactory
from pier.agents.installed.base import BaseInstalledAgent
from pier.environments.base import ExecResult
from pier.environments.factory import EnvironmentFactory
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.trial.execution import TrialExecution


INSTALLED_AGENT_CLASSES = tuple(
    agent_class
    for agent_class in AgentFactory._AGENTS
    if issubclass(agent_class, BaseInstalledAgent)
)
EXPECTED_CA_ENV = {
    "SSL_CERT_FILE": CA_BUNDLE_PATH,
    "REQUESTS_CA_BUNDLE": CA_BUNDLE_PATH,
    "CURL_CA_BUNDLE": CA_BUNDLE_PATH,
    "GIT_SSL_CAINFO": CA_BUNDLE_PATH,
    "NODE_EXTRA_CA_CERTS": EXTRA_CA_PATH,
    "CODEX_CA_CERTIFICATE": EXTRA_CA_PATH,
}


RSA_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIDEzCCAfugAwIBAgIUC6jlOjnFqrdqNzrasZ11JeJ9pvQwDQYJKoZIhvcNAQEL
BQAwGDEWMBQGA1UEAwwNUGllci1UZXN0LVJTQTAgFw0yNjA4MzExMjE0MzBaGA8y
MTI2MDgwNzEyMTQzMFowGDEWMBQGA1UEAwwNUGllci1UZXN0LVJTQTCCASIwDQYJ
KoZIhvcNAQEBBQADggEPADCCAQoCggEBANVlCZS55esuK0B+qDXCTAfgmKtUFrtu
C+xq6zaUTh4qX6R0dpyoKW1yfj2LwgOnArTRevefO5/td/Tnr8piFqy808hf4y5E
+OLbpXCkpntpFF6IRZMhIIe0a2qcJ34Nxg3u+6wJtHSdHif9ZCZ4WM8+JySH144S
cwMt5rY28NF1Vxp1889vX8GK7sBfi/LTUFoJ60R2C+8WivaD+bzTuXjDy4+bemOj
RFNuVQoV9X1D83RiXfQrwNvdhRw+QlgebmZQnpvt/gVDfe/RyyKQy0t987CuF+sG
HrEQqMf8wY6Kwt0+a/azUdT9aFruSQFQlEBjDH9U7R+c9PfUKktafN0CAwEAAaNT
MFEwHQYDVR0OBBYEFBJWr5wLDCjxlYFLQWA3flz4JdQ4MB8GA1UdIwQYMBaAFBJW
r5wLDCjxlYFLQWA3flz4JdQ4MA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQEL
BQADggEBADS/96f2sRHxTKH/1spz2FNbhH+eRzoHb0B3bcpM7SZx3nmA3dLn0wwJ
l9WqVQRHJULE8PKr8eVCcpootI1YrDgyuKMHZV7N95kNaDiGbPYNFPTnObzfpW9q
7MtogdVX3JsopQO3WXIXo5Alsj/u2kUOxQE0VDosMhlniYTuu29tWgybO9BTUmwe
x5Ls7xA27DTN0lLwly3uBxa3kvJitQiV7bc/0qV7b6Jr3p50EwC/V5VJwZGoH5mL
l0K8lDAXrXyXosCOqRnIO78pZLGvHBPgDxKvym/luSUR9YxsxyHCWALxZgirjdw3
LV2nJsX9QsV8Gq9UnNUgDlWa+ISmiSw=
-----END CERTIFICATE-----
"""
EC_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIBhTCCASugAwIBAgIUe1ktbzVL6SlEYsCOCUSC2cdvk9kwCgYIKoZIzj0EAwIw
FzEVMBMGA1UEAwwMUGllci1UZXN0LUVDMCAXDTI2MDgzMTEyMTQzMFoYDzIxMjYw
ODA3MTIxNDMwWjAXMRUwEwYDVQQDDAxQaWVyLVRlc3QtRUMwWTATBgcqhkjOPQIB
BggqhkjOPQMBBwNCAATHHgz7eNJ67+pyKuj0LkhO6yGVSC8ojW/UPhOQKxmnEE8a
5Z1dKVNKah8sJHXiQ6/zGQadYlZSMt/lUo9YRtXyo1MwUTAdBgNVHQ4EFgQUmLyE
2AaH7LfE+SrsXOcCiXZ2ntcwHwYDVR0jBBgwFoAUmLyE2AaH7LfE+SrsXOcCiXZ2
ntcwDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAgNIADBFAiB3CgXJc3J0GPPz
UXm8mlpJ7PD9BSSpmoiFVit+cAST4AIhAP9cqDHvmerwyjMJEXH0peFfzqGOlr9v
IM1olitdZ2gF
-----END CERTIFICATE-----
"""


@pytest.fixture
def certificates() -> tuple[str, str]:
    rsa_certificate = x509.load_pem_x509_certificate(RSA_CERTIFICATE.encode("ascii"))
    ec_certificate = x509.load_pem_x509_certificate(EC_CERTIFICATE.encode("ascii"))
    assert isinstance(rsa_certificate.public_key(), rsa.RSAPublicKey)
    assert isinstance(ec_certificate.public_key(), ec.EllipticCurvePublicKey)
    assert ec_certificate.public_key().curve.name == "secp256r1"

    blocks = tuple(
        split_pem_certificates(pem)[0] for pem in (RSA_CERTIFICATE, EC_CERTIFICATE)
    )
    assert len(blocks) == 2
    assert blocks[0] != blocks[1]
    return blocks


class FakeInstalledAgent(BaseInstalledAgent):
    def __init__(
        self,
        logs_dir: Path,
        spec: AgentInstallSpec | None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._spec: Any = spec
        super().__init__(logs_dir=logs_dir, version="1.0", extra_env=extra_env)

    @staticmethod
    def name() -> str:
        return "fake"

    def install_spec(self) -> AgentInstallSpec:
        return self._spec

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        pass

    def populate_context_post_run(self, context: Any) -> None:
        pass


class RecordingEnvironment:
    def __init__(self) -> None:
        self.exec_calls: list[dict[str, Any]] = []

    def agent_process_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        return env

    async def exec(self, **kwargs: Any) -> ExecResult:
        self.exec_calls.append(kwargs)
        return ExecResult(return_code=0)


def install_spec() -> AgentInstallSpec:
    return AgentInstallSpec(
        agent_name="fake",
        version="1.0",
        steps=[InstallStep(run="install agent", user="agent")],
    )


def corrupt_certificate(certificate: str) -> str:
    lines = certificate.splitlines()
    return f"{lines[0]}\nTm90IGEgY2VydGlmaWNhdGU=\n{lines[-1]}\n"


def make_registered_agent(
    agent_class: type[BaseInstalledAgent], logs_dir: Path
) -> BaseInstalledAgent:
    return agent_class(logs_dir=logs_dir, model_name="test/model")


@pytest.mark.parametrize(
    "agent_class",
    INSTALLED_AGENT_CLASSES,
    ids=lambda agent_class: agent_class.__name__,
)
def test_every_registered_installed_agent_adds_ca_step(
    agent_class: type[BaseInstalledAgent],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    monkeypatch.delenv(EXTRA_CA_CERTS_ENV, raising=False)
    unconfigured = make_registered_agent(agent_class, tmp_path / "plain")
    unconfigured_spec = unconfigured.resolved_install_spec()

    ca_path = tmp_path / "extra.pem"
    ca_path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(ca_path))
    configured = make_registered_agent(agent_class, tmp_path / "configured")
    configured_spec = configured.resolved_install_spec()

    assert unconfigured_spec is not None
    assert configured_spec is not None
    assert configured_spec.steps[:-1] == unconfigured_spec.steps
    assert len(configured_spec.steps) == len(unconfigured_spec.steps) + 1

    ca_steps = [step for step in configured_spec.steps if EXTRA_CA_PATH in step.run]
    assert len(ca_steps) == 1
    assert configured_spec.steps[-1] is ca_steps[0]
    assert ca_steps[0].user == "root"
    assert f"> {EXTRA_CA_PATH}" in ca_steps[0].run
    assert f": > {CA_BUNDLE_PATH}" in ca_steps[0].run
    assert configured_spec.fingerprint() != unconfigured_spec.fingerprint()


@pytest.mark.parametrize(
    "agent_class",
    INSTALLED_AGENT_CLASSES,
    ids=lambda agent_class: agent_class.__name__,
)
@pytest.mark.asyncio
async def test_every_registered_installed_agent_exec_gets_ca_env(
    agent_class: type[BaseInstalledAgent],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    ca_path = tmp_path / "extra.pem"
    ca_path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(ca_path))
    agent = make_registered_agent(agent_class, tmp_path / "agent")
    environment = RecordingEnvironment()

    await agent._exec(environment, "echo probe")

    assert environment.exec_calls[-1]["env"] == EXPECTED_CA_ENV


@pytest.mark.parametrize(
    "agent_class",
    INSTALLED_AGENT_CLASSES,
    ids=lambda agent_class: agent_class.__name__,
)
def test_registered_installed_agents_do_not_override_ca_hooks(
    agent_class: type[BaseInstalledAgent],
) -> None:
    overrides = [
        method_name
        for method_name in ("_exec", "resolved_install_spec")
        if method_name in agent_class.__dict__
    ]

    assert not overrides, (
        f"{agent_class.__name__} overrides {overrides}; registered installed agents "
        "must use BaseInstalledAgent's shared CA injection hooks."
    )


def test_installed_agent_modules_do_not_bypass_exec_ca_injection() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    installed_dir = repo_root / "src/pier/agents/installed"
    offenders = [
        path.relative_to(repo_root).as_posix()
        for path in installed_dir.rglob("*.py")
        if path.name != "base.py"
        and re.search(r"\.exec\(", path.read_text(encoding="utf-8"))
    ]

    assert not offenders, (
        "Direct .exec(...) calls in installed-agent modules bypass BaseInstalledAgent "
        "CA injection; route commands through exec_as_agent/exec_as_root instead. "
        f"Offending files: {offenders}"
    )


@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_not_configured_returns_defaults_and_preserves_spec(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(EXTRA_CA_CERTS_ENV, raising=False)
    else:
        monkeypatch.setenv(EXTRA_CA_CERTS_ENV, value)

    spec = install_spec()
    resolved = with_extra_ca_certs(spec)

    assert ca_trust_env() == {}
    assert read_extra_ca_certs() is None
    assert resolved.steps == spec.steps
    assert resolved.fingerprint() == spec.fingerprint()


def test_ca_trust_env_separates_replace_and_additive_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    path = tmp_path / "extra.pem"
    path.write_text("".join(certificates), encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    env = ca_trust_env()
    replace_vars = {
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "GIT_SSL_CAINFO",
    }
    additive_vars = {"NODE_EXTRA_CA_CERTS", "CODEX_CA_CERTIFICATE"}

    assert len(env) == 6
    assert set(env) == replace_vars | additive_vars
    assert replace_vars.isdisjoint(additive_vars)
    assert all(env[key] == CA_BUNDLE_PATH for key in replace_vars)
    assert all(env[key] == EXTRA_CA_PATH for key in additive_vars)


def test_split_pem_certificates_ignores_non_certificate_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    first, second = certificates
    crl = "-----BEGIN X509 CRL-----\nignored crl\n-----END X509 CRL-----\n"
    private_key = (
        "-----BEGIN PRIVATE KEY-----\nignored key\n-----END PRIVATE KEY-----\n"
    )
    bundle = (
        "leading junk\n"
        + crl
        + first
        + "junk between blocks\n"
        + private_key
        + second
        + "trailing junk\n"
    )

    assert split_pem_certificates(first) == [first]
    assert split_pem_certificates(first + second) == [first, second]
    assert split_pem_certificates(bundle) == [first, second]
    assert split_pem_certificates("") == []

    path = tmp_path / "bundle.pem"
    path.write_text(bundle, encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))
    assert read_extra_ca_certs() == bundle


def test_read_extra_ca_certs_expands_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    pem = "".join(certificates)
    path = tmp_path / "extra.pem"
    path.write_text(pem, encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, "~/extra.pem")

    assert read_extra_ca_certs() == pem


@pytest.mark.parametrize(
    "contents", ["not a PEM bundle", "-----BEGIN PRIVATE KEY-----\n"]
)
def test_invalid_extra_ca_certs_raise_with_env_and_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "invalid.pem"
    path.write_text(contents, encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError) as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message
    # A file with no certificate delimiters at all is the one case that should
    # still get the generic message; the truncation and TRUSTED-CERTIFICATE
    # paths have their own, more specific errors.
    assert "contains no '-----BEGIN CERTIFICATE-----' block" in message


def test_missing_extra_ca_certs_raise_with_env_and_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "missing.pem"
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError) as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message


def test_valid_then_truncated_certificate_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    path = tmp_path / "truncated-bundle.pem"
    path.write_text(
        certificates[0] + "-----BEGIN CERTIFICATE-----\ntruncated body\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError) as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message
    assert "has 2" in message
    assert "only 1 closed" in message


def test_single_unterminated_certificate_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "unterminated.pem"
    path.write_text("-----BEGIN CERTIFICATE-----\ntruncated body\n", encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError) as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message
    # The file visibly holds a BEGIN marker, so "contains no BEGIN CERTIFICATE
    # block" would send the reader looking in the wrong place.
    assert "has 1 '-----BEGIN CERTIFICATE-----' block(s) but only 0 closed" in message


def test_trusted_certificate_only_is_rejected_as_trusted_certificate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "trusted-only.pem"
    path.write_text(
        "-----BEGIN TRUSTED CERTIFICATE-----\ntrusted body\n"
        "-----END TRUSTED CERTIFICATE-----\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError) as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message
    # Naming the actual problem beats the generic "no plain certificate" error.
    assert "-----BEGIN TRUSTED CERTIFICATE-----" in message
    assert "convert it first" in message


def test_trusted_certificate_after_plain_certificate_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    path = tmp_path / "mixed-trusted.pem"
    path.write_text(
        certificates[0] + "-----BEGIN TRUSTED CERTIFICATE-----\ntrusted body\n"
        "-----END TRUSTED CERTIFICATE-----\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError, match="TRUSTED CERTIFICATE") as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message
    assert "openssl x509 -in in.pem -out out.pem" in message


def test_corrupt_certificate_body_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    path = tmp_path / "corrupt.pem"
    path.write_text(corrupt_certificate(certificates[0]), encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError) as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message
    assert "certificate #1" in message


def test_corrupt_certificate_reports_its_bundle_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    path = tmp_path / "corrupt-bundle.pem"
    path.write_text(
        certificates[0] + corrupt_certificate(certificates[1]), encoding="utf-8"
    )
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError) as exc_info:
        read_extra_ca_certs()

    message = str(exc_info.value)
    assert EXTRA_CA_CERTS_ENV in message
    assert str(path) in message
    assert "certificate #2" in message
    assert "certificate #1" not in message


def test_two_valid_certificates_still_read_successfully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    pem = "".join(certificates)
    path = tmp_path / "valid-bundle.pem"
    path.write_text(pem, encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    assert read_extra_ca_certs() == pem


def test_with_extra_ca_certs_rejects_corrupt_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    path = tmp_path / "corrupt.pem"
    path.write_text(corrupt_certificate(certificates[0]), encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    with pytest.raises(ExtraCaCertsError, match="certificate #1"):
        with_extra_ca_certs(install_spec())


def test_with_extra_ca_certs_appends_root_step_without_mutating_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    pem = "".join(certificates)
    path = tmp_path / "extra.pem"
    path.write_text(pem, encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))

    spec = install_spec()
    original = spec.model_dump()
    original_steps = list(spec.steps)
    augmented = with_extra_ca_certs(spec)

    assert augmented is not spec
    assert spec.model_dump() == original
    assert spec.steps == original_steps
    assert augmented.steps[:-1] == original_steps
    assert len(augmented.steps) == len(original_steps) + 1
    assert augmented.steps[-1].user == "root"
    assert augmented.steps[-1].run == ca_trust_install_script(pem)
    assert augmented.fingerprint() != spec.fingerprint()


def test_different_ca_files_change_install_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    first_path = tmp_path / "first.pem"
    second_path = tmp_path / "second.pem"
    first_path.write_text(certificates[0], encoding="utf-8")
    second_path.write_text(certificates[1], encoding="utf-8")
    spec = install_spec()

    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(first_path))
    first = with_extra_ca_certs(spec)
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(second_path))
    second = with_extra_ca_certs(spec)

    assert first.fingerprint() != spec.fingerprint()
    assert second.fingerprint() != spec.fingerprint()
    assert first.fingerprint() != second.fingerprint()


def test_cache_key_is_extended_by_ca_digest_without_populating_new_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    first_path = tmp_path / "first.pem"
    second_path = tmp_path / "second.pem"
    first_path.write_text(certificates[0], encoding="utf-8")
    second_path.write_text(certificates[1], encoding="utf-8")
    fixed = AgentInstallSpec(
        agent_name="fake",
        cache_key="fixed-key",
        steps=[InstallStep(run="install agent")],
    )
    no_key = install_spec()

    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(first_path))
    first = with_extra_ca_certs(fixed)
    # The digest covers the generated install step, not just the PEM, so a change
    # to ca_trust_install_script() also invalidates a fixed cache key.
    first_digest = hashlib.sha256(
        ca_trust_install_script(certificates[0]).encode("utf-8")
    ).hexdigest()[:16]
    assert first.cache_key == f"fixed-key-ca-{first_digest}"
    assert first.fingerprint() != fixed.fingerprint()
    assert fixed.cache_key == "fixed-key"

    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(second_path))
    second = with_extra_ca_certs(fixed)
    assert second.cache_key != first.cache_key

    unchanged = with_extra_ca_certs(no_key)
    assert no_key.cache_key is None
    assert unchanged.cache_key is None


def test_install_script_payloads_round_trip_and_write_expected_files(
    certificates: tuple[str, str],
) -> None:
    pem = "".join(certificates)
    script = ca_trust_install_script(pem)
    payloads = re.findall(r"printf '%s' '([^']+)' \| base64 -d", script)
    decoded = [base64.b64decode(payload).decode("utf-8") for payload in payloads]

    assert decoded[0] == pem
    assert set(decoded[1:]) == set(certificates)
    assert all(decoded[1:].count(certificate) >= 1 for certificate in certificates)
    assert f"> {EXTRA_CA_PATH}" in script
    assert f": > {CA_BUNDLE_PATH}" in script
    assert f"cat {EXTRA_CA_PATH} >> {CA_BUNDLE_PATH}" in script

    anchor_paths = re.findall(r"\$anchor_dir/(pier-extra-ca-\d+\.crt)", script)
    assert {f"pier-extra-ca-{index}.crt" for index in range(1, 3)} == set(anchor_paths)


def test_install_script_probes_every_system_bundle_without_short_circuit(
    certificates: tuple[str, str],
) -> None:
    script = ca_trust_install_script("".join(certificates))
    start = script.index("for bundle in ")
    end = script.index("\n\n# certifi", start)
    system_loop = script[start:end]

    # These are container paths, so execute-shape coverage is more reliable than
    # running the script on the host. The union contract requires no short circuit.
    assert all(f'"{bundle}"' in system_loop for bundle in SYSTEM_CA_BUNDLES)
    assert "break" not in system_loop


def test_install_script_probes_certifi_with_load_bearing_quote_rules(
    certificates: tuple[str, str],
) -> None:
    script = ca_trust_install_script("".join(certificates))
    start = script.index("for py in ")
    end = script.index("\n\n# Nothing public", start)
    python_loop = script[start:end]

    for candidate in CERTIFI_PYTHONS:
        if "*" in candidate:
            assert candidate in python_loop
            assert f'"{candidate}"' not in python_loop
        else:
            assert f'"{candidate}"' in python_loop


def test_base_installed_resolved_install_spec_delegates_to_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    ca_path = tmp_path / "extra.pem"
    ca_path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(ca_path))
    base_resolver = BaseAgent.resolved_install_spec
    calls: list[BaseAgent] = []

    def observed_resolver(agent: BaseAgent):
        calls.append(agent)
        return base_resolver(agent)

    monkeypatch.setattr(BaseAgent, "resolved_install_spec", observed_resolver)
    agent = FakeInstalledAgent(tmp_path, install_spec())

    resolved = agent.resolved_install_spec()

    assert calls == [agent]
    assert resolved.steps[-1].user == "root"
    assert EXTRA_CA_PATH in resolved.steps[-1].run


def test_resolved_install_spec_handles_none_and_applies_ca(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    monkeypatch.delenv(EXTRA_CA_CERTS_ENV, raising=False)
    base_agent = SimpleNamespace(install_spec=lambda: None)
    assert BaseAgent.resolved_install_spec(base_agent) is None

    path = tmp_path / "extra.pem"
    path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))
    spec = install_spec()
    resolved = FakeInstalledAgent(tmp_path, spec).resolved_install_spec()

    assert resolved is not None
    assert resolved.steps[:-1] == spec.steps
    assert resolved.steps[-1].user == "root"


@pytest.mark.asyncio
async def test_install_uses_resolved_spec_and_runs_ca_step_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    path = tmp_path / "extra.pem"
    path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))
    agent = FakeInstalledAgent(tmp_path, install_spec())
    environment = RecordingEnvironment()

    await agent.install(environment)

    assert len(environment.exec_calls) == 2
    assert environment.exec_calls[0]["command"] == "set -o pipefail; install agent"
    assert environment.exec_calls[-1]["user"] == "root"
    assert environment.exec_calls[-1]["command"].startswith("set -o pipefail; set -eu")
    assert CA_BUNDLE_PATH in environment.exec_calls[-1]["command"]


def test_trial_environment_wiring_uses_resolved_install_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved_spec = object()
    allowlist = object()
    calls: list[str] = []
    captured: dict[str, Any] = {}

    agent = SimpleNamespace(
        resolved_install_spec=lambda: calls.append("resolved") or resolved_spec,
        network_allowlist=lambda: allowlist,
    )
    task_environment = object()
    task = SimpleNamespace(
        name="task",
        paths=SimpleNamespace(environment_dir=tmp_path),
        config=SimpleNamespace(
            environment=task_environment,
            agent=SimpleNamespace(user="agent"),
        ),
    )

    def create_environment(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        EnvironmentFactory,
        "create_environment_from_config",
        create_environment,
    )

    result = TrialExecution._create_environment(
        environment_config=object(),
        task=task,
        session_id="session",
        trial_paths=object(),
        logger=logging.getLogger(__name__),
        agent=agent,
    )

    assert result is not None
    assert calls == ["resolved"]
    assert captured["agent_install_spec"] is resolved_spec
    assert captured["network_allowlist"] is allowlist


@pytest.mark.asyncio
async def test_exec_extra_env_overrides_ca_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    path = tmp_path / "extra.pem"
    path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(path))
    agent = FakeInstalledAgent(
        tmp_path,
        install_spec(),
        extra_env={"SSL_CERT_FILE": "agent-choice.pem"},
    )
    environment = RecordingEnvironment()

    await agent._exec(
        environment,
        "echo ok",
        env={"SSL_CERT_FILE": "per-command-choice.pem"},
    )

    expected = ca_trust_env()
    expected["SSL_CERT_FILE"] = "agent-choice.pem"
    assert environment.exec_calls[-1]["env"] == expected


@pytest.mark.asyncio
async def test_exec_preserves_env_without_ca_or_extra_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(EXTRA_CA_CERTS_ENV, raising=False)
    agent = FakeInstalledAgent(tmp_path, install_spec())
    environment = RecordingEnvironment()
    supplied = {"KEEP": "this"}

    await agent._exec(environment, "echo supplied", env=supplied)
    await agent._exec(environment, "echo none")

    assert environment.exec_calls[-2]["env"] is supplied
    assert environment.exec_calls[-1]["env"] is None


@pytest.mark.parametrize(
    "agent_class",
    INSTALLED_AGENT_CLASSES,
    ids=lambda agent_class: agent_class.__name__,
)
@pytest.mark.asyncio
async def test_install_withholds_ca_env_for_every_registered_agent(
    agent_class: type[BaseInstalledAgent],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    ca_path = tmp_path / "extra.pem"
    ca_path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(ca_path))
    agent = make_registered_agent(agent_class, tmp_path / "agent")
    spec = agent.resolved_install_spec()
    environment = RecordingEnvironment()

    assert len(spec.steps) > 1
    await agent.install(environment)

    assert len(environment.exec_calls) == len(spec.steps)
    for call in environment.exec_calls:
        assert set(call["env"] or {}).isdisjoint(EXPECTED_CA_ENV)


@pytest.mark.asyncio
async def test_install_preserves_step_and_agent_env_without_ca_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    ca_path = tmp_path / "extra.pem"
    ca_path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(ca_path))
    spec = AgentInstallSpec(
        agent_name="fake",
        steps=[
            InstallStep(
                user="agent",
                run="first step",
                env={"STEP_KEY": "step-value"},
            ),
            InstallStep(user="root", run="second step"),
        ],
    )
    agent = FakeInstalledAgent(tmp_path, spec, extra_env={"AGENT_KEY": "agent-value"})
    environment = RecordingEnvironment()

    await agent.install(environment)

    assert environment.exec_calls[0]["env"] == {
        "STEP_KEY": "step-value",
        "AGENT_KEY": "agent-value",
    }
    assert all(
        call["env"]["AGENT_KEY"] == "agent-value" for call in environment.exec_calls
    )
    assert all(
        set(call["env"]).isdisjoint(EXPECTED_CA_ENV) for call in environment.exec_calls
    )


@pytest.mark.asyncio
async def test_normal_exec_restores_ca_env_after_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, certificates: tuple[str, str]
) -> None:
    ca_path = tmp_path / "extra.pem"
    ca_path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(ca_path))
    agent = FakeInstalledAgent(tmp_path, install_spec())
    environment = RecordingEnvironment()

    await agent.install(environment)
    await agent._exec(environment, "echo after install")

    assert environment.exec_calls[-1]["env"] == EXPECTED_CA_ENV


@pytest.mark.parametrize("method_name", ["exec_as_root", "exec_as_agent"])
@pytest.mark.asyncio
async def test_exec_helpers_can_suppress_ca_env_or_use_default(
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    certificates: tuple[str, str],
) -> None:
    ca_path = tmp_path / "extra.pem"
    ca_path.write_text(certificates[0], encoding="utf-8")
    monkeypatch.setenv(EXTRA_CA_CERTS_ENV, str(ca_path))
    agent = FakeInstalledAgent(tmp_path, install_spec())
    environment = RecordingEnvironment()
    exec_helper = getattr(agent, method_name)

    await exec_helper(environment, "echo without ca", inject_ca_env=False)
    assert environment.exec_calls[-1]["env"] is None

    await exec_helper(environment, "echo with ca")
    assert environment.exec_calls[-1]["env"] == EXPECTED_CA_ENV
