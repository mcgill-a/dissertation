import neptune
from keys.tokens import NEPTUNE_API_TOKEN
from project.parameters import params

# select project
neptune.init('mcgill-a/translation', api_token=NEPTUNE_API_TOKEN)

# create experiment
neptune.create_experiment(name='get-started-example-from-docs',
                          params=params)

# send some metrics
for i in range(1, params['N_EPOCHS']):
    neptune.log_metric('iteration', i)
    neptune.log_metric('loss', 1/i**0.5)
    neptune.log_text('magic values', 'magic value {}'.format(0.95*i**2))

neptune.set_property('model', 'lightGBM')

# send some images
for j in range(0, 5):
    array = np.random.rand(10, 10, 3)*255
    array = np.repeat(array, 30, 0)
    array = np.repeat(array, 30, 1)
    neptune.log_image('mosaics', array)

neptune.stop()