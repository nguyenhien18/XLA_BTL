"""Evaluate the selected fine-tuned model; frozen score/NMS thresholds .55/.30."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import tifffile
from segmentation import StarDistPredictor
from pipeline import load_image
from custom_postprocess import from_maps
from evaluate import overlap_matrix,match_instances,scores,save_csv,plot_comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parents[1]
    model_dir=root/'models/stardist_finetuned_v1'
    metadata=json.loads((model_dir/'training_metadata.json').read_text())
    if metadata['status']!='complete':raise RuntimeError('Training is not complete')
    model=StarDistPredictor(model_dir,20261105).model
    from csbdeep.utils import normalize
    from stardist.nms import non_maximum_suppression_sparse
    from stardist.matching import matching_dataset
    out=root/'results/stardist_finetuned';out.mkdir(exist_ok=True)
    figs=root/'results/figures/finetune';figs.mkdir(exist_ok=True)
    metrics=root/'results/metrics'
    demo=set((root/'data/splits/demo.txt').read_text().split())
    rows=[];validation=[];truths=[];predictions=[];provenance={}
    for n,name in enumerate((root/'data/splits/test.txt').read_text().split(),1):
        raw=load_image(root/f'data/prepared/images/{name}.png')
        prob,dist=model.predict(normalize(raw,1,99.8),show_tile_progress=False)
        pred,detail=from_maps(prob,dist,raw.shape)
        ref=non_maximum_suppression_sparse(detail['distances'],detail['scores'],detail['origins'],nms_thresh=.3)[3]
        if not np.array_equal(ref,detail['kept']):raise RuntimeError('Custom/reference NMS differ on '+name)
        target=out/f'{name}.tif';tifffile.imwrite(target,pred,photometric='minisblack')
        gt=tifffile.imread(root/f'data/prepared/labels/{name}.tif')
        true_ids,pred_ids,iou=overlap_matrix(gt,pred)
        for threshold in [.5,.6,.7,.8]:
            m=match_instances(true_ids,pred_ids,iou,threshold)
            row=dict(image=name,iou_threshold=threshold,ground_truth_count=len(true_ids),predicted_count=len(pred_ids),TP=m['TP'],FP=m['FP'],FN=m['FN'],matched_iou_sum=m['matched_iou_sum'],**scores(m['TP'],m['FP'],m['FN'],m['matched_iou_sum']))
            rows.append(row)
            if name in demo and threshold==.5:plot_comparison(raw,gt,pred,m,row,figs/f'{name}_comparison.png')
        truths.append(gt);predictions.append(pred)
        provenance[name]=dict(input_sha256=hashlib.sha256((root/f'data/prepared/images/{name}.png').read_bytes()).hexdigest(),prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),custom_nms_equals_reference=True)
        print(f'Fine-tuned test {n}/82: {name}, GT={len(true_ids)}, pred={len(pred_ids)}',flush=True)
    summaries=[]
    for threshold in [.5,.6,.7,.8]:
        selected=[r for r in rows if r['iou_threshold']==threshold]
        tp,fp,fn=[sum(r[k] for r in selected) for k in ['TP','FP','FN']]
        summed_iou=sum(r['matched_iou_sum'] for r in selected)
        summary=dict(model='fine_tuned',images=len(selected),iou_threshold=threshold,TP=tp,FP=fp,FN=fn,**scores(tp,fp,fn,summed_iou),macro_F1=float(np.mean([r['F1'] for r in selected])))
        summaries.append(summary)
        check=matching_dataset(truths,predictions,thresh=threshold,show_progress=False)
        assert (tp,fp,fn)==(check.tp,check.fp,check.fn)
        assert np.isclose(summary['F1'],check.f1)
        validation.append(dict(iou_threshold=threshold,reference_counts_equal=True,reference_F1_equal=True))
    save_csv(metrics/'finetuned_test_per_image.csv',rows)
    save_csv(metrics/'finetuned_test_summary.csv',summaries)
    before=list(csv.DictReader((metrics/'stardist_python_test_optimized_summary.csv').open(encoding='utf-8-sig')))
    comparison=[]
    for old,new in zip(before,summaries):
        assert float(old['iou_threshold'])==new['iou_threshold']
        comparison.append(dict(iou_threshold=new['iou_threshold'],pretrained_F1=float(old['F1']),finetuned_F1=new['F1'],delta_F1=new['F1']-float(old['F1']),pretrained_precision=float(old['precision']),finetuned_precision=new['precision'],pretrained_recall=float(old['recall']),finetuned_recall=new['recall']))
    save_csv(metrics/'finetune_before_after.csv',comparison)
    (metrics/'finetune_evaluation_validation.json').write_text(json.dumps(dict(reference_checks=validation,images=len(provenance),weights_sha256=metadata['weights_sha256'],thresholds={'prob':.55,'nms':.3},custom_code_sha256=hashlib.sha256((root/'src/custom_postprocess.py').read_bytes()).hexdigest(),provenance=provenance),indent=2))
    history=json.loads((model_dir/'history.json').read_text())
    epochs=np.arange(1,len(history['loss'])+1)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    for ax,key,title in zip(axes,['loss','prob_loss','dist_loss'],['Total loss','Probability loss','Radial-distance loss']):
        ax.plot(epochs,history[key],label='Training (augmented patches)')
        ax.plot(epochs,history['val_'+key],label='Validation (fixed patches)')
        ax.axvline(metadata['best_epoch'],color='gray',ls='--',lw=1)
        ax.set(xlabel='Epoch',ylabel='Loss',title=title);ax.legend(fontsize=7)
    fig.suptitle('Actual fine-tuning history; dashed line = selected checkpoint')
    fig.savefig(figs/'training_loss.png',dpi=180);fig.savefig(figs/'training_loss.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.plot([r['iou_threshold'] for r in comparison],[r['pretrained_F1'] for r in comparison],'o-',label='Pretrained')
    ax.plot([r['iou_threshold'] for r in comparison],[r['finetuned_F1'] for r in comparison],'s-',label='Fine-tuned')
    ax.set(xlabel='Evaluation IoU threshold',ylabel='Pooled instance F1',ylim=(0,1),title='Same 82 test images; score=0.55, NMS=0.30');ax.legend()
    fig.savefig(figs/'before_after.png',dpi=180);fig.savefig(figs/'before_after.svg');plt.close(fig)
    report=['# Kết quả fine-tune StarDist','',f"Đã huấn luyện tiếp từ pretrained trên 381 ảnh train, dùng 82 ảnh validation. Chạy {metadata['epochs_completed']} epoch; chọn epoch {metadata['best_epoch']} theo val_loss thấp nhất. Thời gian {metadata['elapsed_seconds']/60:.1f} phút trên CPU.",'','Đánh giá đủ 82 ảnh test sau khi chốt checkpoint. So sánh giữ nguyên score 0,55 và NMS 0,30; chưa tối ưu lại ngưỡng cho mô hình fine-tune. Phần dựng đa giác/NMS tự cài đã khớp tập ứng viên của thư viện trên tất cả ảnh đánh giá. Các chỉ số được đối chiếu hàm tham chiếu StarDist.','','| IoU | F1 pretrained | F1 fine-tune | Chênh lệch |','|---|---:|---:|---:|']
    for r in comparison:report.append(f"| {r['iou_threshold']} | {r['pretrained_F1']:.4f} | {r['finetuned_F1']:.4f} | {r['delta_F1']:+.4f} |")
    report += ['','Fine-tune không đảm bảo cải thiện mọi chỉ số. Đường loss dùng các patch và augmentation nên không đồng nhất với đánh giá toàn ảnh. Mô hình pretrained có thể đã thấy một phần DSB2018 trong huấn luyện gốc; fine-tune không loại bỏ giới hạn này.','','Hồ sơ: configs/finetune.json; models/stardist_finetuned_v1/history.csv, history.json, training_metadata.json; results/metrics/finetune_before_after.csv; results/figures/finetune/.']
    (root/'docs/FINETUNE_RESULTS.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(comparison),flush=True)

if __name__=='__main__':main()
