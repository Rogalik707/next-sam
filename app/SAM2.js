import path from "path";

import * as ort from "onnxruntime-web/all";
import pako from "pako";
// Принудительно используем CPU
ort.env.wasm.numThreads = 4; // Можно настроить количество потоков
ort.env.wasm.simd = true; // Включаем SIMD для лучшей производительности

// Полифилл для Float16Array
class Float16Array {
  constructor(buffer) {
    this.buffer = buffer;
    this.length = buffer.byteLength / 2;
  }

  static fromBuffer(buffer) {
    const view = new DataView(buffer);
    const float32 = new Float32Array(buffer.byteLength / 2);
    for (let i = 0; i < float32.length; i++) {
      const uint16 = view.getUint16(i * 2, true);
      const sign = (uint16 & 0x8000) >> 15;
      const exponent = (uint16 & 0x7C00) >> 10;
      const fraction = uint16 & 0x03FF;

      if (exponent === 0) {
        float32[i] = (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 0x0400);
      } else if (exponent === 0x1F) {
        float32[i] = fraction ? NaN : (sign ? -Infinity : Infinity);
      } else {
        float32[i] = (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 0x0400);
      }
    }
    return float32;
  }
}

const DECODER_URL =
  "https://huggingface.co/flyvi/sam2.1/resolve/main/sam2.1_hiera_tiny_decoder.onnx";

export class SAM2 {
  bufferDecoder = null;
  sessionDecoder = null;
  image_encoded = null;

  constructor() {}

  async downloadModels() {
    this.bufferDecoder = await this.downloadModel(DECODER_URL);
  }

  async downloadModel(url) {
    const filename = url.split('/').pop();
    
    // Попытка получить из IndexedDB
    try {
      const cached = await this.getFromIndexedDB(filename);
      if (cached) {
        return cached;
      }
    } catch (e) {
      console.warn("IndexedDB read failed:", e);
    }

    // Загрузка если нет в кэше
    console.log("File not in cache, downloading from " + url);
    let buffer = null;
    try {
      buffer = await fetch(url, {
        headers: new Headers({
          Origin: location.origin,
        }),
        mode: "cors",
      }).then((response) => response.arrayBuffer());
    } catch (e) {
      console.error("Download of " + url + " failed: ", e);
      return null;
    }

    // Сохранение в IndexedDB
    try {
      await this.saveToIndexedDB(filename, buffer);
      console.log("Stored " + filename);
    } catch (e) {
      console.warn("IndexedDB write failed:", e);
    }

    return buffer;
  }

  async getFromIndexedDB(filename) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("SAM2Models", 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(["models"], "readonly");
        const store = transaction.objectStore("models");
        const getRequest = store.get(filename);

        getRequest.onsuccess = () => resolve(getRequest.result);
        getRequest.onerror = () => reject(getRequest.error);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains("models")) {
          db.createObjectStore("models");
        }
      };
    });
  }

  async saveToIndexedDB(filename, buffer) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("SAM2Models", 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const transaction = db.transaction(["models"], "readwrite");
        const store = transaction.objectStore("models");
        const putRequest = store.put(buffer, filename);

        putRequest.onsuccess = () => resolve();
        putRequest.onerror = () => reject(putRequest.error);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains("models")) {
          db.createObjectStore("models");
        }
      };
    });
  }

  async createSessions() {
    const success = await this.getDecoderSession();
    return {
      success: success,
      device: success ? this.sessionDecoder[1] : null,
    };
  }

  async getORTSession(model) {
    // Пробуем разные провайдеры в порядке приоритета
    const providers = [
      'webgl',
      'wasm',
      'cpu'
    ];

    try {
      // Сначала пробуем WebGL для лучшей производительности
      if (providers.includes('webgl')) {
        try {
          const session = await ort.InferenceSession.create(model, {
            executionProviders: ['webgl'],
            graphOptimizationLevel: 'all'
          });
          console.log("Using WebGL provider");
          return [session, 'webgl'];
        } catch (e) {
          console.warn("WebGL initialization failed:", e);
        }
      }

      // Затем пробуем WASM с SIMD если доступно
      if (providers.includes('wasm')) {
        try {
          const session = await ort.InferenceSession.create(model, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            extra: {
              wasm: {
                numThreads: 4,
                simd: true
              }
            }
          });
          console.log("Using WASM provider with SIMD");
          return [session, 'wasm'];
        } catch (e) {
          console.warn("WASM initialization failed:", e);
        }
      }

      // В крайнем случае используем CPU
      const session = await ort.InferenceSession.create(model, {
        executionProviders: ['cpu'],
        graphOptimizationLevel: 'all'
      });
      console.log("Using CPU provider");
      return [session, 'cpu'];

    } catch (e) {
      console.error("Failed to create session with any provider:", e);
      return null;
    }
  }

  async getDecoderSession() {
    if (!this.sessionDecoder) {
      const result = await this.getORTSession(this.bufferDecoder);
      if (!result) {
        throw new Error("Failed to initialize decoder session with any provider");
      }
      this.sessionDecoder = result;
    }
    return this.sessionDecoder;
  }

  setImageEmbeddings(embeddings) {
    // Функция для декодирования сжатых данных
    const decodeEmbedding = (embedding) => {
        // Декодируем base64
        const binaryString = atob(embedding.data);
        
        // Конвертируем строку в Uint8Array
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        // Распаковываем zlib
        const decompressed = pako.inflate(bytes);
        
        // Конвертируем в Float32Array (из float16)
        const float32Data = Float16Array.fromBuffer(decompressed.buffer);
        
        return new ort.Tensor("float32", float32Data, embedding.dims);
    };

    this.image_encoded = {
        high_res_feats_0: decodeEmbedding(embeddings.high_res_feats_0),
        high_res_feats_1: decodeEmbedding(embeddings.high_res_feats_1),
        image_embed: decodeEmbedding(embeddings.image_embed)
    };
  }

  async decode(points, masks) {
    const [session, device] = await this.getDecoderSession();

    const flatPoints = points.map((point) => {
      return [point.x, point.y];
    });

    const flatLabels = points.map((point) => {
      return point.label;
    });

    console.log({
      flatPoints,
      flatLabels,
      masks
    });

    let mask_input, has_mask_input
    if (masks) {
      mask_input = masks
      has_mask_input = new ort.Tensor("float32", [1], [1])
    } else {
      // dummy data
      mask_input = new ort.Tensor(
        "float32",
        new Float32Array(256 * 256),
        [1, 1, 256, 256]
      )
      has_mask_input = new ort.Tensor("float32", [0], [1])
    }

    const inputs = {
      image_embed: this.image_encoded.image_embed,
      high_res_feats_0: this.image_encoded.high_res_feats_0,
      high_res_feats_1: this.image_encoded.high_res_feats_1,
      point_coords: new ort.Tensor("float32", flatPoints.flat(), [
        1,
        flatPoints.length,
        2,
      ]),
      point_labels: new ort.Tensor("float32", flatLabels, [
        1,
        flatLabels.length,
      ]),
      mask_input: mask_input,
      has_mask_input: has_mask_input,
    };

    return await session.run(inputs);
  }
}
