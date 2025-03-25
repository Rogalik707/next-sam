import { SAM2 } from "./SAM2";
import { Tensor } from "onnxruntime-web";

const sam = new SAM2();
const SERVER_URL = '/api';

const stats = {
  device: "unknown",
  downloadModelsTime: [],
  decodeTimes: [],
  lastError: null
};

self.onmessage = async (e) => {
  try {
    const { type, data } = e.data;

    if (type === "ping") {
      self.postMessage({ type: "downloadInProgress" });
      const startTime = performance.now();
      
      try {
        await sam.downloadModels();
      } catch (error) {
        console.error("Failed to download decoder model:", error);
        throw new Error("Failed to download decoder model");
      }
      
      const durationMs = performance.now() - startTime;
      stats.downloadModelsTime.push(durationMs);

      self.postMessage({ type: "loadingInProgress" });
      const report = await sam.createSessions();

      if (!report.success) {
        throw new Error("Failed to create decoder session");
      }

      stats.device = report.device;
      stats.lastError = null;

      self.postMessage({ type: "pong", data: report });
      self.postMessage({ type: "stats", data: stats });

    } else if (type === "encodeImage") {
      try {
        const { imageData } = data;
        
        console.time('encode-total');
        
        // Создаем Blob из исходных данных изображения
        const imageBlob = new Blob([imageData], { type: 'image/jpeg' });
        
        // Создаем FormData
        const formData = new FormData();
        formData.append('image', imageBlob, 'image.jpg');

        // Отправляем запрос через прокси
        console.time('fetch-request');
        const response = await fetch(`${SERVER_URL}/segmentation_embeddings`, {
          method: 'POST',
          body: formData,
          headers: {
            'APIKEY': 'qwerty'
          }
        });
        console.timeEnd('fetch-request');

        console.time('json-parse');
        const embeddings = await response.json();  // Теперь response.json() сразу дает эмбеддинги
        console.timeEnd('json-parse');
        
        if (!embeddings || !embeddings.high_res_feats_0 || !embeddings.high_res_feats_1 || !embeddings.image_embed) {
          throw new Error("Invalid embeddings format received from server");
        }

        // Устанавливаем эмбеддинги
        console.time('set-embeddings');
        sam.setImageEmbeddings(embeddings);
        console.timeEnd('set-embeddings');
        
        console.timeEnd('encode-total');
        
        self.postMessage({
          type: "encodeImageDone"
        });

      } catch (error) {
        console.error("Image encoding failed:", error);
        throw new Error("Failed to encode image on server");
      }

    } else if (type === "setImageEmbeddings") {
      const { embeddings } = data;
      
      if (!embeddings || !embeddings.high_res_feats_0 || !embeddings.high_res_feats_1 || !embeddings.image_embed) {
        throw new Error("Invalid embeddings format received from server");
      }

      try {
        sam.setImageEmbeddings(embeddings);
        stats.lastError = null;
        
        self.postMessage({
          type: "setImageEmbeddingsDone"
        });
      } catch (error) {
        console.error("Failed to set image embeddings:", error);
        throw new Error("Failed to process server embeddings");
      }

    } else if (type === "decodeMask") {
      const {points, maskArray, maskShape} = data;

      if (!points || !points.length) {
        throw new Error("No points provided for mask generation");
      }

      const startTime = performance.now();

      let decodingResults;
      try {
        if (maskArray) {
          const maskTensor = new Tensor("float32", maskArray, maskShape);
          decodingResults = await sam.decode(points, maskTensor);
        } else {
          decodingResults = await sam.decode(points);
        }

        const durationMs = performance.now() - startTime;
        stats.decodeTimes.push(durationMs);
        stats.lastError = null;

        self.postMessage({
          type: "decodeMaskResult",
          data: decodingResults,
          stats: { durationMs }
        });
      } catch (error) {
        console.error("Mask decoding failed:", error);
        throw new Error("Failed to generate mask");
      }

    } else if (type === "stats") {
      self.postMessage({ type: "stats", data: stats });
    } else {
      throw new Error(`Unknown message type: ${type}`);
    }

  } catch (error) {
    stats.lastError = error.message;
    self.postMessage({ 
      type: "error", 
      error: error.message,
      stats: stats
    });
  }
};
