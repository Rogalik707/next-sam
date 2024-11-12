"use client"

import React, { useState, useEffect, useRef, createContext, useCallback } from 'react';

// UI
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { LoaderCircle } from 'lucide-react'

// Image manipulations
import { resizeCanvas, mergeMasks, canvasToFloat32Array, sliceTensorMask } from "@/lib/imageutils"

export default function Home() {
  // state
  const [loading, setLoading] = useState(false)
  const [samWorkerReady, setSamWorkerReady] = useState(false)
  const [imageEncoded, setImageEncoded] = useState(false)

  // web worker, image and mask
  const samWorker = useRef(null)
  const [image, setImage] = useState(null)    // canvas
  const [mask, setMask] = useState(null)    // canvas
  const [imageURL, setImageURL] = useState("/photo.png")
  const canvasEl = useRef(null)

  // Start encoding image
  const encodeImageClick = async () => {
    const canvas = canvasEl.current
    const float32Data = canvasToFloat32Array(resizeCanvas(canvas, {w: 1024, h: 1024}))

    samWorker.current.postMessage({ type: 'encodeImage', data: float32Data });   
    setLoading(true)
  }

  // Start decoding, prompt with mouse coords
  const imageClick = (event) => {
    if (!imageEncoded) return;

    const canvas = canvasEl.current
    const rect = event.target.getBoundingClientRect();

    // input image will be resized to 1024x1024 -> also normalize pos to 1024x1024
    const point = {
      x: (event.clientX - rect.left) / canvas.width * 1024,
      y: (event.clientY - rect.top) / canvas.height * 1024,
      label: 1
    }

    samWorker.current.postMessage({ type: 'decodeMask', data: point });   
    setLoading(true)
  }

  // Decoding finished -> parse result and update mask
  const handleDecodingResults = (decodingResults) => {
    // SAM2 returns 3 mask along with scores -> select best one    
    const maskTensors = decodingResults.masks
    const maskScores = decodingResults.iou_predictions.cpuData
    const bestMaskIdx = maskScores.indexOf(Math.max(...maskScores))
    const maskCanvas = sliceTensorMask(maskTensors, bestMaskIdx)    

    setMask((prevMask) => {
      if (prevMask) {
        return mergeMasks(maskCanvas, prevMask)
      } else {
        return resizeCanvas(maskCanvas, {w: canvasEl.current.width, h: canvasEl.current.height})
      }
    })
    setLoading(false)
  }

  // Handle web worker messages
  const onWorkerMessage = (event) => {
    const {type, data} = event.data

    if (type == "pong" ) {
      setLoading(false)
      setSamWorkerReady(true)
    } else if (type == "encodeImageDone" ) {
      setLoading(false)
      setImageEncoded(true)
    } else if (type == "decodeMaskResult" ) {
      handleDecodingResults(data) 
    }
  }

  // Load web worker 
  useEffect(() => {
    if (!samWorker.current) {
      samWorker.current = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      samWorker.current.addEventListener('message', onWorkerMessage)
      samWorker.current.postMessage({ type: 'ping' });   

      setLoading(true)
    }
  }, [onWorkerMessage, handleDecodingResults])

  // Load image, store in offscreen canvas
  useEffect(() => {
    if (imageURL) {
      const img = new Image();
      img.src = imageURL
      img.onload = function() {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        canvas.getContext('2d').drawImage(img, 0, 0)
        setImage(canvas)
      }
    }
  }, [imageURL]);

  // Offscreen canvas changed, draw it 
  useEffect(() => {
    if (image) {
      const canvas = canvasEl.current
      const ctx = canvas.getContext('2d');
      canvas.width = 512
      canvas.height = 512
      ctx.drawImage(image, 0, 0, image.width, image.height,
        0, 0, canvas.width, canvas.height);      
    }
  }, [image]);

  // Mask changed, draw original image again and mask on top with some alpha
  useEffect(() => {
    if (mask) {
      const canvas = canvasEl.current
      const ctx = canvas.getContext('2d')

      ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvas.width, canvas.height);      
      ctx.globalAlpha = 0.4
      canvas.getContext('2d').drawImage(mask, 0, 0);
      ctx.globalAlpha = 1;
    }
  }, [mask, image])

  return (
    <div className="flex items-center justify-center min-h-screen bg-background p-4">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle>Next/SAM2 - Image Segmentation in the browser with onnxruntime-web and Meta's SAM2</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <Button 
              onClick={encodeImageClick}
              disabled={loading || (samWorkerReady && imageEncoded)}
            >
              <p className="flex items-center gap-2">
              { loading &&
                  <LoaderCircle className="animate-spin w-6 h-6" />
              }
              { loading && !samWorkerReady && "Loading model"}
              { !loading && samWorkerReady && !imageEncoded && "Encode image"}
              { loading && samWorkerReady && !imageEncoded && "Encoding"}
              { !loading && samWorkerReady && imageEncoded && "Ready. Click on image"}
              { loading && samWorkerReady && imageEncoded && "Decoding"}
              </p>
            </Button>
            <canvas ref={canvasEl} onClick={imageClick}/>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
