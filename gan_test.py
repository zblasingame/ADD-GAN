import utils.file_ops as fops
import models.gan
import json

model = models.gan.GAN(
    num_features=12, num_epochs=1000, normalize=True,
    debug=True, latent_vector_size=100,
    batch_size=1000, ns_param=0., adpt_l=0,
    res_depth=1, dr_param=1, batch_param=0.,
    display_step=10, learning_rate=0.005,
    reg_param=0.01
)

print('hola')

exploit = 'freak'
# exploit = 'nginx_keyleak'
# exploit = 'nginx_rootdir'

data = []

for i in range(5):
    trX, trY = fops.load_data(
        (
            './data/three-step/{}/subset_{}/train_set.csv'
        ).format(exploit, i)
    )

    model.train(trX, trY)

    for i in range(5):
        teX, teY = fops.load_data(
            (
                './data/three-step/{}/subset_{}/test_set.csv'
            ).format(exploit, i)
        )

        data.append(model.test(teX, teY))

with open('results/{}.json'.format(exploit), 'w') as f:
    json.dump(data, f, indent=2)
