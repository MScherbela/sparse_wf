# %%
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

default_section = ws.Section(
    name="Default Section",
    panels=[
        wr.LinePlot(x="opt/step", y=["opt/E_smooth"]),
        wr.LinePlot(
            x="opt/step",
            y=["opt/E_std"],
            log_y=True,
            ignore_outliers=True,
            smoothing_type="exponentialTimeWeighted",
            smoothing_factor=0.8,
            range_y=(0.1, 2),
        ),
        wr.LinePlot(
            x="pretrain/step",
            y=["pretrain/loss"],
            log_y=True,
            smoothing_type="exponentialTimeWeighted",
            smoothing_factor=0.4,
        ),
        wr.LinePlot(x="opt/step", y=["opt/t_step"], ignore_outliers=True, range_y=(0, 10)),
        wr.LinePlot(x="opt/step", y=["opt/log10_S_cond_nr"], ignore_outliers=True),
        wr.LinePlot(x="opt/step", y=["mcmc/mean_cluster_size"]),
    ],
    is_open=True,
    layout_settings=ws.SectionLayoutSettings(columns=3, rows=2),
    panel_settings=ws.SectionPanelSettings(x_axis="opt/step")
)

workspace = ws.Workspace(
    entity="tum_daml_nicholas",
    project="N2",
    name="Default Views",
    sections=[default_section],
    settings=ws.WorkspaceSettings(max_runs=25),
)
workspace.save()
