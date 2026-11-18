"""v14: restore the injected audit helper in the expanded main namespace."""
from flashrsi_harnessrisk_supervisor_hermes_workspace_v13 import (
    source, case_tool_environment, native_context_profile,
)
from flashrsi_harnessrisk_opencode_mock_identity_v10 import mock_client_environment

source = source.replace('private-v1-hermes-contextfit-v13-native-prune',
                        'private-v1-hermes-contextfit-v14-native-prune')
exec(compile(source, __file__, 'exec'))
