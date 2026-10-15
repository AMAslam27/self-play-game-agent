import pytest

from evaluation import match_plots as plots
from tests.unit_tests.common.mock_utils import FakeFigure, FakePath


@pytest.fixture
def plot_environment(monkeypatch):
    figure = FakeFigure()
    canvases = []
    paths = []

    def figure_factory(**kwargs):
        return figure

    def path_factory(path):
        result = FakePath(path)
        paths.append(result)
        return result

    monkeypatch.setattr(plots, "Figure", figure_factory)
    monkeypatch.setattr(plots, "FigureCanvasAgg", canvases.append)
    monkeypatch.setattr(plots, "Path", path_factory)
    return plots, figure, canvases, paths
