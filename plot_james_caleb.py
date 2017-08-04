import utils.file_ops as fops
import models.gan
import json
import numpy as np
import tensorflow as tf
import os
import plotly.plotly as py
import plotly.graph_objs as go

hyperparameters = dict(
    num_features=12, num_epochs=1000, normalize=True,
    debug=True, latent_vector_size=72,
    batch_size=1000, ns_param=.5, adpt_l=0,
    res_depth=1, dr_param=1, batch_param=1e-2,
    display_step=10, d_learning_rate=1e-3,
    reg_param=1e-3, g_learning_rate=1e-4
)

model = models.gan.GAN(**hyperparameters)

data = []

for i in range(5):
    trX, trY = fops.load_data(
        (
            './data/three-step/caleb/subset_{}/train_set.csv'
        ).format(i)
    )

    model.train(trX, trY)

    for j in range(5):
        teX, teY = fops.load_data(
            (
                './data/three-step/caleb/subset_{}/test_set.csv'
            ).format(j)
        )

        d = model.test(teX, teY)
        data.append(d)

# plot data
accs = [entry['accuracy'] for entry in data]

fig = go.Figure(
    data=[go.Box(
        x=accs,
        name='caleb',
        boxmean='sd'
    )],
    layout=go.Layout(
        title='Classifier Accuracy of Linear Encryption Batches',
        xaxis=dict(title='Accuracy (%)')
    )
)

py.plot(fig, filename='add-gan-james-caleb-results-box')
