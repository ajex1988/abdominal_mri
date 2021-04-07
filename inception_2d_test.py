import tensorflow as tf
import inception_2d
import cv2
import numpy as np

def model_fn(features,labels,mode,params):
    global_step = tf.train.get_global_step()
    input_layer = features['MRI']
    logits, end_points= inception_2d.inception_v1_caffe(input_layer,
                                                     num_classes=params['n_classes'],
                                                     spatial_squeeze=True,
                                                     dropout_keep_prob=0.6)

    # predict mode
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class':predicted_classes,
            'prob':tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes)
        my_acc = tf.reduce_mean(tf.cast(tf.equal(labels, predicted_classes), tf.float32))
        tf.summary.scalar('accuracy',my_acc)


    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    # hook
    train_hook_list = []
    train_tensors_log = {'accuracy':accuracy[1],
                         'my_acc': my_acc,
                         'loss':loss,
                         'global_step':global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log,every_n_iter=200))
    #train_hook_list.append(tf_debug.LocalCLIDebugHook(ui_type="readline"))
    # training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9)
        optimizer = tf.train.AdagradOptimizer(0.001)
        #optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=0.001,weight_decay=0.0002,momentum=0.9)
        train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,training_hooks=train_hook_list)
    # compute evaluation metrics
    eval_metric_ops = {
        'accuracy':tf.metrics.accuracy(labels=labels,predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)


def run_test():
    print("Classifying the test_mri.png using a pretrained Google Inception 2D model.")
    # Test image file
    img_file = 'test_mri.png'
    # Load the gray-scale image. We use the cv2.IMREAD_GRAYSCALE tag to force 1-channel load
    # since our trained model takes 1-channel image as input
    mri_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    mri_img = mri_img.astype(np.float32)
    mri_img -= 37.0
    dsize = (224, 224)
    mri_img = cv2.resize(mri_img, dsize)
    mri_img = np.expand_dims(mri_img, 2)
    mri_img = np.expand_dims(mri_img, 0)
    my_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"MRI": mri_img},
        shuffle=False,
        batch_size=1)

    model_folder = '0621'
    feature_columns = [tf.feature_column.numeric_column(key='MRI', shape=(224, 224, 1))]
    n_classes = 30
    classifier = tf.estimator.Estimator(
        model_dir=model_folder,
        model_fn=model_fn,
        params={
            'feature_columns': feature_columns,
            'n_classes': n_classes
        }
    )
    predictions = classifier.predict(input_fn=my_input_fn)
    for result in predictions:
        print(result)
        prediction_file = 'test_mri_predicted.txt'
        with open(prediction_file, 'w') as f:
            f.write(result['prob'])


if __name__ == "__main__":
    run_test()