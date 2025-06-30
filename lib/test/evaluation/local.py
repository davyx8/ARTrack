from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/workspace/ARTrack/data/got10k_lmdb'
    settings.got10k_path = '/workspace/ARTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/workspace/ARTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/workspace/ARTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/workspace/ARTrack/data/lasot_lmdb'
    settings.lasot_path = '/workspace/ARTrack/data/lasot'
    settings.network_path = '/workspace/ARTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/workspace/ARTrack/data/nfs'
    settings.otb_path = '/workspace/ARTrack/data/otb'
    settings.prj_dir = '/workspace/ARTrack'
    settings.result_plot_path = '/workspace/ARTrack/output/test/result_plots'
    settings.results_path = '/workspace/ARTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/workspace/ARTrack/output'
    settings.segmentation_path = '/workspace/ARTrack/output/test/segmentation_results'
    settings.tc128_path = '/workspace/ARTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/workspace/ARTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/workspace/ARTrack/data/trackingnet'
    settings.uav_path = '/workspace/ARTrack/data/uav'
    settings.vot18_path = '/workspace/ARTrack/data/vot2018'
    settings.vot22_path = '/workspace/ARTrack/data/vot2022'
    settings.vot_path = '/workspace/ARTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

