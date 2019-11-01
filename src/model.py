from fastai import *
from fastai.vision import *

path = Path('../data/')
tfms = get_transforms(flip_vert=True)

np.random.seed(352)
data = ImageDataBunch.from_folder(path, valid_pct=0.2, ds_tfms=tfms, size=224).normalize(imagenet_stats)
data.show_batch(3, figsize=(15, 11))

# create a learner based on a pretrained densenet 121 model
learn = cnn_learner(data, models.densenet121, metrics=error_rate)

# use the learning rate finder to find the optimal learning rate
learn.lr_find()
learn.recorder.plot()

lr = 1e-2 # learning rate choosen based on the result of the learning rate finder
# train for 5 epochs
learn.fit_one_cycle(5, slice(lr))

# save the model
learn.save('stage-1-dn121')

# unfreeze and finetune
learn.load('stage-1-dn121');
learn.unfreeze()
learn.lr_find()

# use the learning rate finder again
learn.recorder.plot()

learn.fit_one_cycle(10, slice(1e-4, lr/10))
learn.save('stage-2-dn121')

# export as pickle file for deployment
learn.export('dn121.pkl')

# model interpretation 
interp = ClassificationInterpretation.from_learner(learn)

# plot images where the model did not perform well
interp.plot_top_losses(4)
# plot confusion matrix
interp.plot_confusion_matrix(dpi=130)
