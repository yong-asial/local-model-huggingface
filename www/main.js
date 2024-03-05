import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.1';

// global var
let sentimentPipeline = null;
let textClassificationPipeline = null;
let startTime;
const resultEl = document.getElementById('result');
const loadingImage = document.getElementById('loading');

const updateResult = (html) => {
    resultEl.innerHTML = html;
    console.log(html); // debug purpose
};

const start = () => {
    loadingImage.style.display = 'block'; // show loading image
    updateResult('Loading ...');
    startTime = performance.now(); // calculate processing time
}

const predict = async (type, pipelineModel, sentence) => {
    if (!sentence) {
        loadingImage.style.display = 'none';
        alert('Input some query');
        return;
    }
    if (!pipelineModel) {
        alert('The model is not loaded.');
        loadingImage.style.display = 'none';
        return;
    }
    updateResult('Processing ...');
    try {
        // process sentence and return result from model pipeline function
        const result = await pipelineModel(sentence);
        const endTime = performance.now(); // calculate processing time
        if (result) {
            updateResult(`
                <p>${JSON.stringify(result)}</p>
                <p>Time taken: ${((endTime - startTime) / 1000).toFixed(2)} seconds (${type}).</p>
            `);
        }
    } catch (e) {
        updateResult(e.toString());
    }
    loadingImage.style.display = 'none'; // hide loading image
};

const getBrowserCachePipeline = async (useRemote, taskName, modelName) => {
    let piplelineModel;
    env.allowRemoteModels = useRemote; // true: use HuggingFace cloud model. false: use downloaded model in browser cache
    try {
        piplelineModel = await pipeline(
            taskName,
            modelName
        );
    } catch (e) {
        console.log(e);
        piplelineModel = null;
    }
    return piplelineModel;
};

const useBrowserCache = async (pipelineCache, taskName, modelName) => {
    env.useBrowserCache = true;
    if (!pipelineCache) {
        updateResult('Loading model from cache...');
        pipelineCache = await getBrowserCachePipeline(false, taskName, modelName);
        if (!pipelineCache) {
            updateResult('Cache not available. Loading remote model...');
            pipelineCache = await getBrowserCachePipeline(true, taskName, modelName);
        }
        updateResult('Model is loaded.');
    }
    await predict('browser', pipelineCache, document.getElementById("query").value);
    return pipelineCache;
};

const useLocalModel = async (pipelineCache, taskName, modelName) => {
    env.localModelPath = './models/';
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.useBrowserCache = false;
    if (!pipelineCache) {
        updateResult('Loading model from local...');
        pipelineCache = await pipeline(
            taskName,
            modelName
        );
        updateResult('Model is loaded.');
    }
    await predict('local', pipelineCache, document.getElementById("query").value);
    return pipelineCache;
};

const sentimentClassifyLocalModel = async () => {
    start();
    // this model doesn't have onnx model. so we use the converter tool to convert from pytorch model to onnx and upload to www/local folder.
    textClassificationPipeline = await useLocalModel(
        textClassificationPipeline,
        'text-classification',
        'local/nlptown/bert-base-multilingual-uncased-sentiment'
    );
};

const sentimentClassify = async () => {
    start();
    // this model has onnx model file. so we can use it directly from HuggingFace. It will first download it then cache it.
    sentimentPipeline = await useBrowserCache(
        sentimentPipeline,
        'sentiment-analysis',
        'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
    );
};

const main = () => {
    document.getElementById("btnSentimentCache").addEventListener('click', sentimentClassify, false);
    document.getElementById("btnSentimenLocal").addEventListener('click', sentimentClassifyLocalModel, false);
};


main();