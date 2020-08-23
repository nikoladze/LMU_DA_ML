def set_tf_nthreads(nthreads):
    # these settings perform better for CPU training at CIP
    # for the particular model we have here
    import os
    os.environ["OMP_NUM_THREADS"] = str(nthreads)
    os.environ["MKL_NUM_THREADS"] = str(nthreads)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(nthreads)
    tf.config.threading.set_inter_op_parallelism_threads(nthreads)
