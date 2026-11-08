#@ org.scijava.Context context

// Open this file with Fiji's Script Editor, choose Groovy and press Run.
// Uses the installed StarDist/CSBDeep/TensorFlow Java plugins, not Python.
import net.imagej.ImageJ
import de.csbdresden.stardist.StarDist2D
import de.csbdresden.stardist.Candidates
import ij.IJ
import ij.io.FileSaver
import ij.io.RoiEncoder
import net.imglib2.img.display.imagej.ImageJFunctions
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream
import java.util.zip.ZipEntry
import java.nio.file.Files

def root = new File('D:/BTL_XLA')
def imageDir = new File(root, 'data/prepared/images')
def outputDir = new File(root, 'results/stardist_fiji')
def metricsDir = new File(root, 'results/metrics')
def splitFile = new File(root, 'data/splits/demo.txt')
def model = 'Versatile (fluorescent nuclei)'
double probability = 0.48
double overlap = 0.30
double low = 1.0
double high = 99.8
int boundary = 0
int tiles = 1

if (!splitFile.isFile()) throw new IOException('Missing demo list: ' + splitFile)
outputDir.mkdirs()
metricsDir.mkdirs()
def names = splitFile.readLines('UTF-8').collect { it.trim() }.findAll { !it.isEmpty() }
if (names.isEmpty() || names.unique(false).size() != names.size()) {
    throw new IOException('Demo list is empty or contains duplicate names')
}
names.each { name ->
    if (!(name ==~ /img_\d{4}/)) throw new IOException('Invalid image name: ' + name)
    if (!new File(imageDir, name + '.png').isFile()) throw new IOException('Missing input: ' + name)
}

def ij2 = new ImageJ(context)
def runId = 'fiji_batch_' + new Date().format('yyyyMMdd_HHmmss') + '_' + UUID.randomUUID().toString().substring(0, 6)
def runLog = new File(metricsDir, runId + '.csv')
def parametersFile = new File(root, 'fiji/parameters.csv')
def csv = { values -> values.collect { '"' + (it == null ? '' : it.toString()).replace('"', '""') + '"' }.join(',') + '\n' }
runLog.setText(csv(['image', 'status', 'objects', 'elapsed_seconds', 'model', 'probability', 'overlap', 'percentile_low', 'percentile_high', 'boundary_exclusion', 'tiles', 'message']), 'UTF-8')
if (!parametersFile.exists()) {
    parametersFile.parentFile.mkdirs()
    parametersFile.setText('run_id,image,model,probability_threshold,nms_threshold,normalization,plugin_version,output,notes\n', 'UTF-8')
}

def labelCount = { image ->
    if (image.getBitDepth() != 16 || image.getStackSize() != 1) {
        throw new IOException('Expected a single 16-bit label image')
    }
    def ids = new HashSet<Integer>()
    def ip = image.getProcessor()
    for (int i = 0; i < ip.getPixelCount(); i++) {
        int value = ip.get(i)
        if (value > 0) ids.add(value)
    }
    return ids.size()
}
def roiCount = { file ->
    int count = 0
    new ZipFile(file).withCloseable { archive ->
        archive.entries().each { entry ->
            if (!entry.isDirectory() && entry.name.toLowerCase().endsWith('.roi')) {
                archive.getInputStream(entry).withCloseable { stream ->
                    byte[] data = stream.bytes
                    if (data.length < 4 || new String(data, 0, 4, 'US-ASCII') != 'Iout') {
                        throw new IOException('Invalid ROI entry: ' + entry.name)
                    }
                }
                count++
            }
        }
    }
    return count
}

int processed = 0
int skipped = 0
int failed = 0
println('StarDist Fiji batch: ' + names.size() + ' demo images')
println('Output folder: ' + outputDir)

names.eachWithIndex { name, index ->
    def labelFile = new File(outputDir, name + '.tif')
    def roiFile = new File(outputDir, name + '_rois.zip')
    def labelTemp = new File(outputDir, '.' + name + '_' + runId + '.part.tif')
    def roiTemp = new File(outputDir, '.' + name + '_' + runId + '.part.zip')
    long start = System.nanoTime()
    def labelImage = null
    def existing = null
    try {
        println('[' + (index + 1) + '/' + names.size() + '] ' + name)
        if (labelFile.exists() || roiFile.exists()) {
            if (!(labelFile.isFile() && roiFile.isFile())) {
                throw new IOException('Only one output exists. Resolve the incomplete pair before retrying; no output overwritten.')
            }
            existing = IJ.openImage(labelFile.absolutePath)
            if (existing == null) throw new IOException('Cannot read existing TIFF')
            int count = labelCount(existing)
            if (roiCount(roiFile) != count) throw new IOException('Existing TIFF and ROI counts differ')
            def input = IJ.openImage(new File(imageDir, name + '.png').absolutePath)
            if (input == null || input.width != existing.width || input.height != existing.height) {
                throw new IOException('Existing TIFF dimensions do not match input')
            }
            input.close()
            skipped++
            println('  SKIP: complete existing pair (' + count + ' objects)')
            runLog.append(csv([name, 'SKIPPED_EXISTING', count, '', model, probability, overlap, low, high, boundary, tiles, 'Existing manual or batch output; its run configuration is not inferred']), 'UTF-8')
        } else {
            def input = ij2.scifio().datasetIO().open(new File(imageDir, name + '.png').absolutePath)
            Map<String, Object> options = [
                input: input, modelChoice: model, normalizeInput: true,
                percentileBottom: low, percentileTop: high,
                probThresh: probability, nmsThresh: overlap,
                outputType: 'Label Image', nTiles: tiles, excludeBoundary: boundary,
                roiPosition: 'Automatic', verbose: false,
                showCsbdeepProgress: false, showProbAndDist: true
            ]
            // process=false avoids opening dialogs or showing intermediate output windows.
            def module = ij2.command().run(StarDist2D.class, false, options).get()
            def labels = module.getOutput('label')
            def prob = module.getOutput('prob')
            def dist = module.getOutput('dist')
            if (labels == null || prob == null || dist == null) throw new IOException('StarDist did not return its outputs; see Fiji console')
            labelImage = ImageJFunctions.wrap(labels.getImgPlus(), name)
            int count = labelCount(labelImage)

            // Reuse the same neural-network predictions and native StarDist NMS code
            // to export the original polygons without changing the user's ROI Manager.
            def polygons = new Candidates(prob.getImgPlus(), dist.getImgPlus(), probability, boundary, null)
            polygons.nms(overlap)
            def winners = polygons.getWinner()
            if (winners.size() != count) throw new IOException('ROI/label count mismatch; no final files saved')
            def checkImage = IJ.createImage('check', '16-bit black', labelImage.width, labelImage.height, 1)
            def checkProcessor = checkImage.getProcessor()
            for (int i = winners.size() - 1; i >= 0; i--) {
                checkProcessor.setColor(i + 1)
                checkProcessor.fill(polygons.getPolygonRoi(winners.get(i)))
            }
            def actualProcessor = labelImage.getProcessor()
            for (int i = 0; i < checkProcessor.getPixelCount(); i++) {
                if (checkProcessor.get(i) != actualProcessor.get(i)) {
                    throw new IOException('Native ROI polygons do not reproduce label pixels')
                }
            }
            checkImage.close()
            if (!new FileSaver(labelImage).saveAsTiff(labelTemp.absolutePath)) throw new IOException('TIFF save failed')
            new ZipOutputStream(new FileOutputStream(roiTemp)).withCloseable { archive ->
                winners.eachWithIndex { candidate, roiIndex ->
                    def roi = polygons.getPolygonRoi(candidate)
                    roi.setName(String.format('roi_%04d', roiIndex + 1))
                    byte[] bytes = RoiEncoder.saveAsByteArray(roi)
                    if (bytes == null) throw new IOException('ROI encoding failed')
                    archive.putNextEntry(new ZipEntry(roi.getName() + '.roi'))
                    archive.write(bytes)
                    archive.closeEntry()
                }
            }
            if (roiCount(roiTemp) != count) throw new IOException('Saved ZIP count mismatch')
            Files.move(labelTemp.toPath(), labelFile.toPath())
            try {
                Files.move(roiTemp.toPath(), roiFile.toPath())
            } catch (Exception moveError) {
                Files.move(labelFile.toPath(), labelTemp.toPath())
                throw moveError
            }
            double elapsed = (System.nanoTime() - start) / 1.0e9
            processed++
            runLog.append(csv([name, 'OK', count, String.format(java.util.Locale.US, '%.3f', elapsed), model, probability, overlap, low, high, boundary, tiles, 'Time includes inference and ROI export; not a pure inference benchmark']), 'UTF-8')
            def version = StarDist2D.class.getPackage().getImplementationVersion() ?: '0.3.0 (installed jar)'
            parametersFile.append(csv([runId, name, model, probability, overlap, 'percentile 1/99.8', version, labelFile.name + ';' + roiFile.name, 'normalize=true;tiles=1;boundary=0;ROI polygons verified against label pixels']), 'UTF-8')
            println('  OK: ' + count + ' objects; TIFF + ROI ZIP saved')
        }
    } catch (Exception error) {
        failed++
        runLog.append(csv([name, 'ERROR', '', '', model, probability, overlap, low, high, boundary, tiles, error.toString()]), 'UTF-8')
        System.err.println('  ERROR: ' + error)
        error.printStackTrace()
    } finally {
        if (labelImage != null) labelImage.close()
        if (existing != null) existing.close()
    }
}
println('DONE: processed=' + processed + ', skipped=' + skipped + ', failed=' + failed)
println('Run log: ' + runLog)
if (failed > 0) throw new IOException('Some images failed. Read the run log; valid existing pairs are preserved on retry.')
