class ModelInfo:

    mrt_model = {'t1_tse_tra_fs_Becken_0008': '0008',
                 't1_tse_tra_fs_Becken_Motion_0010': '0010',
                 't1_tse_tra_fs_mbh_Leber_0004': '0004',
                 't1_tse_tra_fs_mbh_Leber_Motion_0005': '0005',
                 't1_tse_tra_Kopf_0002': '0002',
                 't1_tse_tra_Kopf_Motion_0003': '0003', 't2_tse_tra_fs_Becken_0009': '0009',
                 't2_tse_tra_fs_Becken_Motion_0011': '0011',
                 't2_tse_tra_fs_Becken_Shim_xz_0012': '0012',
                 't2_tse_tra_fs_navi_Leber_0006': '0006',
                 't2_tse_tra_fs_navi_Leber_Shim_xz_0007': '0007'}

    mrt_smodel = {'t1_tse_tra_fs_Becken_0008': 'Becken', 't1_tse_tra_fs_Becken_Motion_0010': 'Becken',
                       't1_tse_tra_fs_mbh_Leber_0004': 'Leber', 't1_tse_tra_fs_mbh_Leber_Motion_0005': 'Leber',
                       't1_tse_tra_Kopf_0002': 'Kopf', 't1_tse_tra_Kopf_Motion_0003': 'Kopf',
                       't2_tse_tra_fs_Becken_0009': 'Becken', 't2_tse_tra_fs_Becken_Motion_0011': 'Becken',
                       't2_tse_tra_fs_Becken_Shim_xz_0012': 'Becken', 't2_tse_tra_fs_navi_Leber_0006': 'Leber',
                       't2_tse_tra_fs_navi_Leber_Shim_xz_0007': 'Leber'}

    mrt_artefact = {'t1_tse_tra_fs_Becken_0008': '', 't1_tse_tra_fs_Becken_Motion_0010': 'Move',
                         't1_tse_tra_fs_mbh_Leber_0004': '', 't1_tse_tra_fs_mbh_Leber_Motion_0005': 'Move',
                         't1_tse_tra_Kopf_0002': '', 't1_tse_tra_Kopf_Motion_0003': 'Move',
                         't2_tse_tra_fs_Becken_0009': '', 't2_tse_tra_fs_Becken_Motion_0011': 'Move',
                         't2_tse_tra_fs_Becken_Shim_xz_0012': 'Shim', 't2_tse_tra_fs_navi_Leber_0006': '',
                         't2_tse_tra_fs_navi_Leber_Shim_xz_0007': 'Shim'}

    def getModelInfo(self, sModel):

        dInfo = {}
        dInfo['sRef'] = mrt_model[]