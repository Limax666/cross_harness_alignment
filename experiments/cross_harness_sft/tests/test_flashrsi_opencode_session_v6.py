"""Regression coverage for the immutable OpenCode v6 protocol wrappers."""


def test_v6_wrappers_emit_only_the_registered_v6_protocol_label():
    import flashrsi_harnessrisk_host_opencode_session_v6 as host
    import flashrsi_harnessrisk_opencode_session_supervisor_v6 as supervisor
    import flashrsi_matrix_opencode_session_v6_20261115 as launcher

    for source in (host.source, supervisor.source):
        assert "private-v1-opencode-native-session-v4-mime" not in source
        assert "private-v1-opencode-native-session-v5-compaction" not in source
        assert "private-v1-opencode-native-session-v6-compaction" in source

    assert "flashrsi_harnessrisk_host_opencode_session_v6.py" in launcher.source
