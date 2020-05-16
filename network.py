import tensorflow as tf

def network(U, theta, out_size=None):
    num = tf.shape(U)[0]
    Height = tf.shape(U)[1]
    Width = tf.shape(U)[2]

    theta_mat = tf.reshape(theta, [num, 2, 3])

    if out_size:
        out_H = out_size[0]
        out_W = out_size[1]

        batch_grids = grid_gen(out_H,out_W,theta)

        #complete

def get_pixel_value(img , x, y):

    #should be used for the other dataset
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_index = tf.range(0, batch_size)
    batch_index = tf.reshape(batch_index, [batch_size,1,1])

    b = tf.tile(batch_index, (1,height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def grid_gen(height)