import utils.datasets as ds
import models.gan
import json
import numpy as np
import os

hyperparameters = dict(
    num_features=12, num_epochs=1000, normalize=True,
    debug=True, latent_vector_size=9,
    batch_size=1000, ns_param=0.5, adpt_l=0,
    res_depth=1, dr_param=1, batch_param=1e-2,
    display_step=10, d_learning_rate=1e-3,
    reg_param=1e-3, g_learning_rate=1e-4
)


model = models.gan.GAN(**hyperparameters)


def get_summary(data):
    out = {}

    mats = np.array([entry['confusion_matrix'] for entry in data])

    out['avg_acc'] = np.mean([entry['accuracy'] for entry in data])
    out['std_acc'] = np.std([entry['accuracy'] for entry in data])
    out['avg_mat'] = np.mean(mats, axis=0).tolist()
    out['std_mat'] = np.std(mats, axis=0).tolist()

    return out


dirs = [el for el in os.listdir('results') if 'trial_' in el]
trials = [int(el.split('_')[1]) for el in dirs]
trials.insert(0, -1)
trial = np.max(trials) + 1
print('Trial number: ' + str(trial))
os.mkdir('results/trial_{}'.format(trial))

exploits = ['multi_encrypt', 'freak', 'nginx_keyleak', 'nginx_rootdir']
summaries = {'hyperparameters': hyperparameters}
raw_data = []

for exploit in exploits:
    data = []

    for i in range(5):

        trX, trY = ds.load_data(
            (
                './data/three-step/{}/subset_{}/train_set.csv'
            ).format(exploit, i)
        )

        model.train(trX, trY)

        for i in range(5):
            teX, teY = ds.load_data(
                (
                    './data/three-step/{}/subset_{}/test_set.csv'
                ).format(exploit, i)
            )

            d = model.test(teX, teY)
            data.append(d)
            raw_data.append(d)

    summaries[exploit] = get_summary(data)

    with open('results/trial_{}/{}.json'.format(trial, exploit), 'w') as f:
        json.dump(data, f, indent=2)


summaries['net'] = get_summary(raw_data)

with open('results/trial_{}/summary.json'.format(trial), 'w') as f:
    json.dump(summaries, f, indent=2)

print('Output in results/trial_{}'.format(trial))
