import tensorflow as tf
import os
import utils
import sys
import numpy as np
import pandas as pd

data_dir = sys.argv[1]
embedding_path = sys.argv[1].split('/')
for dir in os.listdir(data_dir):
    print(dir)
    images = utils.load_data([os.path.join(os.path.join(data_dir, dir),img) for img in os.listdir(os.path.join(data_dir,dir))])

    with tf.Graph().as_default():
        with tf.Session() as sess:

            print('Loading feature extraction model')
            utils.load_model(sys.argv[2])
            f=open('node.txt','w')
            for i in [n.name for n in tf.get_default_graph().as_graph_def().node]:
                f.write(i+'\n')
            f.close()
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            print(embeddings,phase_train_placeholder)
            embedding_size = embeddings.get_shape()[1]
            feed_dict = { images_placeholder:images, phase_train_placeholder:False }

            embedding_array = sess.run(embeddings, feed_dict=feed_dict)
            print(embedding_array)

            df = pd.DataFrame(embedding_array)
            print(df.shape)
            df.to_csv('data_embedding_people/'+dir+'_train.csv')
            df.to_csv()


#verify
# Top_same_different = []
# for i in range(0, 221):
#     for j in range(i+1, 222):
#         similarity = verify(embedding_array[i], embedding_array[j])
#         Top_same_different.append(similarity)
# x = sorted(Top_same_different)
# print(x)