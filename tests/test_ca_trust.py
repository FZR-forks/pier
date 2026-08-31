import base64
import logging
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pier.agents.base import BaseAgent
from pier.agents.ca_trust import (
    CA_BUNDLE_PATH,
    EXTRA_CA_CERTS_ENV,
    EXTRA_CA_PATH,
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


@pytest.fixture
def certificates(tmp_path: Path) -> tuple[str, str]:
    paths = (
        Path("/tmp/pukirootca2022rsa.crt"),
        Path("/tmp/pukirootca2022ec.crt"),
    )
    if not all(path.is_file() for path in paths):
        paths = (tmp_path / "rsa.crt", tmp_path / "ec.crt")
        for name, path in zip(("rsa", "ec"), paths, strict=True):
            key_path = tmp_path / f"{name}.key"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(key_path),
                    "-out",
                    str(path),
                    "-days",
                    "1",
                    "-subj",
                    f"/CN={name}",
                ],
                check=True,
                capture_output=True,
            )

    blocks = tuple(
        split_pem_certificates(path.read_text(encoding="utf-8"))[0] for path in paths
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
