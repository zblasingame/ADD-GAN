import plotly.plotly as py
import plotly.graph_objs as go
import json
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    'param_name',
    type=str,
    help='Name of parameter to plot.'
)

args = parser.parse_args()

param_name = args.param_name

exploits = ['caleb', 'freak', 'nginx_keyleak', 'nginx_rootdir']
# exploits = ['freak', 'nginx_keyleak', 'nginx_rootdir']

path = 'results/{}'.format(param_name)

for exploit in exploits:
    data = []
    sum_data = []

    for name in os.listdir(path):
        f_name = '{}/{}/{}.json'.format(path, name, exploit)
        s_name = '{}/{}/summary.json'.format(path, name)

        with open(f_name, 'r') as f:
            data.append(json.load(f))

        with open(s_name, 'r') as f:
            sum_data.append(json.load(f))

    params = [entry['hyperparameters'][param_name] for entry in sum_data]
    params = np.array(params).astype(np.float)
    accs = [[entry['accuracy'] for entry in d] for d in data]
    accs = np.array(accs)

    inds = params.argsort()
    params = params[inds]
    accs = accs[inds]

    boxes = [go.Box(
        y=accs[i],
        name='{} = {:.4e}'.format(param_name, params[i]),
        boxmean='sd'
    ) for i in range(len(params))]

    layout = go.Layout(
        title=('Accuracy vs {} for {}'.format(param_name, exploit)),
        yaxis=dict(title='Accuracy (%)')
    )

    fig = go.Figure(data=boxes, layout=layout)

    py.plot(fig, filename='add-gan-{}-for-{}'.format(param_name, exploit))
