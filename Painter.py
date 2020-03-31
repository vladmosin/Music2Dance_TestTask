"""from Config import Config
from plotly import graph_objects as go
import plotly
from datetime import datetime
from pathlib import Path


def draw(train_rewards, test_rewards, config: Config):
    step = config.max_episodes // len(test_rewards)
    x = list(range(1, config.max_episodes + 2, step))[:len(test_rewards)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=test_rewards, name="Test reward (exploit)"))
    fig.add_trace(go.Scatter(x=x, y=train_rewards, name="Train reward (exploration)"))
    fig.add_trace(
        go.Scatter(x=x, y=[300] * len(test_rewards), line=dict(color="grey", width=2, dash="dash"), name="win"))
    fig.update_layout(
        xaxis=dict(title_text="Episodes"),
        yaxis=dict(title_text="Reward"),
        title_text="TD3: {}".format(config.env_id)
    )
    cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    Path("graphics").mkdir(parents=True, exist_ok=True)
    plotly.offline.plot(fig, filename="graphics/Chart_{}.html".format(cur_time), auto_open=False)
"""