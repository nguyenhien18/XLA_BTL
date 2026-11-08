"""Reproducible CPU fine-tuning of pretrained weights on frozen train/val splits."""
from pathlib import Path
import os,json,time,hashlib,traceback
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import tifffile
from PIL import Image

def main():
    root=Path(__file__).resolve().parents[1]
    protocol_path=root/'configs/finetune.json'
    protocol=json.loads(protocol_path.read_text())
    destination=root/'models'/protocol['model_name']
    if destination.exists(): raise RuntimeError('Refusing to overwrite an existing training run')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(protocol['seed'])
    from stardist.models import StarDist2D, Config2D
    from csbdeep.utils import normalize
    splits={s:(root/f'data/splits/{s}.txt').read_text().split() for s in ['train','val','test']}
    assert all(not (set(splits[a])&set(splits[b])) for a,b in [('train','val'),('train','test'),('val','test')])
    def read(names):
        xs,ys=[],[]
        for name in names:
            with Image.open(root/f'data/prepared/images/{name}.png') as im: x=np.array(im)
            y=tifffile.imread(root/f'data/prepared/labels/{name}.tif')
            assert x.shape==y.shape and min(x.shape)>=128
            xs.append(normalize(x,1,99.8).astype(np.float32));ys.append(y)
        return xs,ys
    train=read(splits['train']); val=read(splits['val'])
    print(f'Loaded train={len(train[0])}, val={len(val[0])}; test images not read for training.',flush=True)
    source=root/'models/2D_versatile_fluo'
    config=json.loads((source/'config.json').read_text())
    config.update(train_epochs=protocol['epochs'],train_steps_per_epoch=protocol['steps_per_epoch'],train_batch_size=protocol['batch_size'],train_patch_size=protocol['patch_size'],train_learning_rate=protocol['learning_rate'],train_n_val_patches=82,train_tensorboard=False,train_reduce_lr={'factor':0.5,'patience':3,'min_delta':0.0001},use_gpu=False)
    model=StarDist2D(Config2D(**config),name=protocol['model_name'],basedir=str(root/'models'))
    model.keras_model.load_weights(str(source/'weights_best.h5'))
    model.prepare_for_training()
    model.callbacks += [tf.keras.callbacks.CSVLogger(str(destination/'history.csv')),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,min_delta=0.0001,restore_best_weights=False)]
    source_hash=hashlib.sha256((source/'weights_best.h5').read_bytes()).hexdigest()
    metadata=dict(protocol=protocol,source_weights_sha256=source_hash,train_images=len(train[0]),validation_images=len(val[0]),selection='lowest validation patch loss; fixed maximum epochs and early stopping; test not used',splits_sha256={s:hashlib.sha256((root/f'data/splits/{s}.txt').read_bytes()).hexdigest() for s in splits},status='running',tensorflow=tf.__version__,device='CPU',caveat='pretrained original training data may overlap DSB2018; not a proven unseen test')
    (destination/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    def augment(x,y):
        k=np.random.randint(4); x=np.rot90(x,k);y=np.rot90(y,k)
        if np.random.rand()<.5:x=np.flip(x,0);y=np.flip(y,0)
        x=x*np.random.uniform(.8,1.2)+np.random.uniform(-.1,.1)
        return np.ascontiguousarray(x,dtype=np.float32),np.ascontiguousarray(y)
    start=time.perf_counter()
    try:
        history=model.train(*train,validation_data=val,augmenter=augment,seed=protocol['seed'])
        metadata.update(status='complete',elapsed_seconds=time.perf_counter()-start,epochs_completed=len(history.history['loss']),best_epoch=int(np.argmin(history.history['val_loss']))+1,best_validation_loss=float(min(history.history['val_loss'])),weights_sha256=hashlib.sha256((destination/'weights_best.h5').read_bytes()).hexdigest())
        (destination/'history.json').write_text(json.dumps({k:[float(x) for x in v] for k,v in history.history.items()},indent=2))
        # Same thresholds as the frozen pretrained optimized comparison.
        (destination/'thresholds.json').write_text(json.dumps({'prob':.55,'nms':.3}))
        print(json.dumps(metadata),flush=True)
    except Exception:
        metadata.update(status='failed',error=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        (destination/'training_metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
