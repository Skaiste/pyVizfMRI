import numpy as np

from ..widgets.HeatMap import HeatMap
from ..dialogs.sw_params_chooser_dialog import SWParamsDialog
from ..widgets.Print3DBrain import Print3DBrain
from ..widgets.Histogram import VectorHeatmap
from ..utils import (FC, phFCD, swFCD)


class ChartFactory:
    @staticmethod
    def compute_local_gbc_vector(data):
        fc = FC.from_fMRI(data.T)
        if np.isscalar(fc) or np.isnan(fc).any():
            return np.array([np.nan])
        n_regions = fc.shape[0]
        fc_wo_diag = fc - np.multiply(fc, np.eye(n_regions))
        return np.mean(fc_wo_diag, axis=1)

    @staticmethod
    def get_chart_data(chart):
        r_min, r_max = chart.get_range(False)
        data = chart.get_data()
        return data[r_min:r_max+1]

    @staticmethod
    def create_heatmap(chart, transformation_function):
        transformed_data = transformation_function(ChartFactory.get_chart_data(chart))
        return HeatMap(transformed_data.T)

    @staticmethod
    def create_fc_heatmap(chart):
        return ChartFactory.create_heatmap(chart, lambda data: FC.from_fMRI(data.T))

    @staticmethod
    def create_sw_heatmap(chart):
        params_dialog = SWParamsDialog(swFCD, ChartFactory.get_chart_data(chart))
        params_dialog.exec()
        return ChartFactory.create_heatmap(chart, lambda data: swFCD.buildFullMatrix(swFCD.from_fMRI(data.T)))

    @staticmethod
    def create_phase_heatmap(chart):
        return ChartFactory.create_heatmap(chart, lambda data: phFCD.buildFullMatrix(phFCD.from_fMRI(data, applyFilters=False)))


    @staticmethod
    def create_gbc_vector_heatmap(chart):
        return VectorHeatmap(ChartFactory.compute_local_gbc_vector(ChartFactory.get_chart_data(chart)))

    @staticmethod
    def create_gbc_3d_brain(chart,default_config):
        return Print3DBrain(ChartFactory.compute_local_gbc_vector(ChartFactory.get_chart_data(chart)),default_config)
