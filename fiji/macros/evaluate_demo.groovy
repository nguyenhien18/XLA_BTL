// Run in Fiji Script Editor (Groovy) or ImageJ2 headless --run.
// Reads ground-truth/predicted label TIFFs; computes all quantities in Java/Groovy.
import ij.IJ
import ij.measure.ResultsTable
import java.awt.GraphicsEnvironment

def root=new File('D:/BTL_XLA')
def names=new File(root,'data/splits/demo.txt').readLines().findAll{it.trim()}
def thresholds=[0.5d,0.6d,0.7d,0.8d]
def out=new File(root,'results/metrics/fiji_quantitative_demo.csv')
def header=['image','iou_threshold','ground_truth_count','predicted_count','TP','FP','FN','precision','recall','F1','binary_iou','binary_dice']
def rows=[]
def table=new ResultsTable()
names.each {name ->
    def gt=IJ.openImage(new File(root,"data/prepared/labels/${name}.tif").path)
    def pred=IJ.openImage(new File(root,"results/stardist_fiji/${name}.tif").path)
    if(gt==null||pred==null)throw new IOException('Cannot open labels: '+name)
    if(gt.width!=pred.width||gt.height!=pred.height)throw new IOException('Shape mismatch: '+name)
    def ga=[:].withDefault{0L};def pa=[:].withDefault{0L};def intersections=[:].withDefault{0L}
    long intersection=0,union=0,gpixels=0,ppixels=0
    def gi=gt.getProcessor();def pi=pred.getProcessor()
    for(int k=0;k<gi.getPixelCount();k++){
        int g=gi.get(k),p=pi.get(k)
        if(g>0){ga[g]++;gpixels++}
        if(p>0){pa[p]++;ppixels++}
        if(g>0||p>0)union++
        if(g>0&&p>0){intersection++;long pair=((long)g<<32)|((long)p&0xffffffffL);intersections[pair]++}
    }
    def gids=ga.keySet().sort();def pids=pa.keySet().sort()
    def gidx=[:];def pidx=[:]
    gids.eachWithIndex{v,i->gidx[v]=i};pids.eachWithIndex{v,i->pidx[v]=i}
    thresholds.each {threshold ->
        def edges=(0..<gids.size()).collect{[]}
        intersections.each {pair,count ->
            int g=(int)(pair>>32),p=(int)(pair&0xffffffffL)
            double iou=count/(double)(ga[g]+pa[p]-count)
            if(iou>=threshold)edges[gidx[g]].add(pidx[p])
        }
        // Augmenting-path bipartite matching maximizes the number of valid pairs.
        int[] owner=new int[pids.size()];Arrays.fill(owner,-1)
        Closure augment
        augment={int g,boolean[] seen ->
            for(int p:edges[g]){
                if(seen[p])continue
                seen[p]=true
                if(owner[p]<0||augment(owner[p],seen)){owner[p]=g;return true}
            }
            return false
        }
        int tp=0
        for(int g=0;g<gids.size();g++)if(augment(g,new boolean[pids.size()]))tp++
        int fp=pids.size()-tp,fn=gids.size()-tp
        double precision=tp+fp>0?tp/(double)(tp+fp):0
        double recall=tp+fn>0?tp/(double)(tp+fn):0
        double f1=2*tp+fp+fn>0?2*tp/(double)(2*tp+fp+fn):1
        def row=[name,threshold,gids.size(),pids.size(),tp,fp,fn,precision,recall,f1,union>0?intersection/(double)union:1,gpixels+ppixels>0?2*intersection/(double)(gpixels+ppixels):1]
        rows.add(row)
        table.incrementCounter();table.addValue('image',name)
        for(int k=1;k<header.size();k++)table.addValue(header[k],(double)row[k])
    }
    gt.close();pred.close()
    println("Evaluated in Fiji: ${name}, GT=${gids.size()}, prediction=${pids.size()}")
}
out.setText(header.join(',')+'\n'+rows.collect{it.join(',')}.join('\n')+'\n','UTF-8')
def summaries=[]
thresholds.each {t ->
    def selected=rows.findAll{it[1]==t}
    long tp=selected.sum{it[4]},fp=selected.sum{it[5]},fn=selected.sum{it[6]}
    summaries.add([t,selected.size(),tp,fp,fn,tp/(double)(tp+fp),tp/(double)(tp+fn),2*tp/(double)(2*tp+fp+fn)])
}
new File(root,'results/metrics/fiji_quantitative_demo_summary.csv').setText('iou_threshold,images,TP,FP,FN,precision,recall,F1\n'+summaries.collect{it.join(',')}.join('\n')+'\n','UTF-8')
if(!GraphicsEnvironment.isHeadless())table.show('Fiji quantitative validation')
println('DONE: '+out.path)
