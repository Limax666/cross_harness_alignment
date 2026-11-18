"""Use the actual prior starting Snapshot, without recycling scores or cases."""
from flashrsi_harnessrisk_algorithm_hermes_v13 import source

source = source.replace('hermes-native-prune-v13', 'hermes-native-prune-v14')
exec(compile(source, __file__, 'exec'))
