/*
 * nodus_onnx_web.js -- Browser ONNX Runtime inference & training client.
 *
 * Loads an .onnx model from the nodus HTTP server and runs inference
 * (and optionally training) in the browser via onnxruntime-web.
 *
 * Usage:
 *   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
 *   <script src="nodus_onnx_web.js"></script>
 *
 *   const client = new NodusOnnxClient("http://localhost:7272");
 *   await client.loadModel(leaseId);
 *   const output = await client.infer(inputFloat32Array, [1, 3, 64, 64]);
 *
 * Training (onnxruntime-web with training bundle):
 *   const loss = await client.trainStep(input, [1,3,64,64], target, [1,10]);
 *
 * onnxruntime-web supports WebAssembly (CPU) and WebGPU backends.
 * Training via onnxruntime-web is experimental and requires the
 * training-enabled WASM build.  If unavailable, trainStep() will
 * throw with a clear message.
 */

// eslint-disable-next-line no-unused-vars
class NodusOnnxClient {
    /**
     * @param {string} serverUrl  Base URL of the nodus server, e.g. "http://localhost:7272"
     */
    constructor(serverUrl) {
        this._base = serverUrl.replace(/\/+$/, "");
        this._session = null;
        this._trainingSession = null;
    }

    /* ------------------------------------------------------------------ */
    /*  Model loading                                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Fetch the ONNX model for a lease and create an inference session.
     *
     * @param {string} leaseId
     * @param {object} [sessionOpts]  ort.InferenceSession.SessionOptions overrides
     * @returns {Promise<void>}
     */
    async loadModel(leaseId, sessionOpts) {
        const url = `${this._base}/api/web/lease/${encodeURIComponent(leaseId)}/weights.onnx`;
        const resp = await fetch(url);
        if (!resp.ok) {
            const body = await resp.text();
            throw new Error(`Failed to fetch ONNX model: ${resp.status} ${body}`);
        }
        const buf = await resp.arrayBuffer();
        this._session = await ort.InferenceSession.create(buf, sessionOpts || {});
    }

    /**
     * Load an ONNX model from raw bytes (ArrayBuffer or Uint8Array).
     *
     * @param {ArrayBuffer|Uint8Array} bytes
     * @param {object} [sessionOpts]
     * @returns {Promise<void>}
     */
    async loadModelFromBytes(bytes, sessionOpts) {
        this._session = await ort.InferenceSession.create(bytes, sessionOpts || {});
    }

    /* ------------------------------------------------------------------ */
    /*  Inference                                                         */
    /* ------------------------------------------------------------------ */

    /**
     * Run inference on a single input tensor.
     *
     * @param {Float32Array} inputData   Contiguous float32 input values
     * @param {number[]}     inputDims   Shape, e.g. [1, 3, 64, 64]
     * @returns {Promise<{data: Float32Array, dims: number[]}>}
     */
    async infer(inputData, inputDims) {
        if (!this._session) throw new Error("No model loaded — call loadModel() first");
        const inputName = this._session.inputNames[0];
        const tensor = new ort.Tensor("float32", inputData, inputDims);
        const feeds = { [inputName]: tensor };
        const results = await this._session.run(feeds);
        const outputName = this._session.outputNames[0];
        const out = results[outputName];
        return { data: out.data, dims: out.dims };
    }

    /* ------------------------------------------------------------------ */
    /*  Training  (requires onnxruntime-web training build)               */
    /* ------------------------------------------------------------------ */

    /**
     * Initialize a training session.  Requires the training-enabled WASM
     * build of onnxruntime-web.
     *
     * @param {string} leaseId
     * @param {object} trainingOpts  { checkpointPath, trainModelPath, optimizerModelPath, evalModelPath }
     *                               — these are URLs or ArrayBuffers.
     * @returns {Promise<void>}
     */
    async initTraining(leaseId, trainingOpts) {
        if (typeof ort.TrainingSession === "undefined") {
            throw new Error(
                "ort.TrainingSession not available — ensure you are using the " +
                "training-enabled build of onnxruntime-web (ort-training-wasm)."
            );
        }
        this._trainingSession = await ort.TrainingSession.create(trainingOpts);
    }

    /**
     * Run a single training step.
     *
     * @param {Float32Array} inputData
     * @param {number[]}     inputDims
     * @param {Float32Array} targetData
     * @param {number[]}     targetDims
     * @returns {Promise<number>}  Scalar loss value
     */
    async trainStep(inputData, inputDims, targetData, targetDims) {
        if (!this._trainingSession) {
            throw new Error("Training session not initialized — call initTraining() first");
        }
        const inputTensor  = new ort.Tensor("float32", inputData, inputDims);
        const targetTensor = new ort.Tensor("float32", targetData, targetDims);
        const feeds = {
            [this._trainingSession.inputNames[0]]: inputTensor,
            [this._trainingSession.inputNames[1]]: targetTensor,
        };
        const results = await this._trainingSession.runTrainStep(feeds);
        const lossName = this._trainingSession.outputNames[0];
        return results[lossName].data[0];
    }

    /**
     * Run the optimizer step to update weights after one or more train steps.
     * @returns {Promise<void>}
     */
    async optimizerStep() {
        if (!this._trainingSession) {
            throw new Error("Training session not initialized");
        }
        await this._trainingSession.runOptimizerStep();
    }

    /**
     * Reset gradients to zero (call between batches).
     * @param {boolean} [setToZero=true]
     * @returns {Promise<void>}
     */
    async resetGrad(setToZero = true) {
        if (!this._trainingSession) {
            throw new Error("Training session not initialized");
        }
        await this._trainingSession.lazyResetGrad();
    }

    /* ------------------------------------------------------------------ */
    /*  Server helpers                                                    */
    /* ------------------------------------------------------------------ */

    /**
     * Trigger ONNX export on the server for a given collection.
     *
     * @param {string} collectionId
     * @param {string} modelClass       e.g. "TinyConvClassifier"
     * @param {number[]} [inputShape]   e.g. [1, 3, 64, 64]
     * @param {number} [opsetVersion]   default 17
     * @returns {Promise<object>}       Server response JSON
     */
    async requestExport(collectionId, modelClass, inputShape, opsetVersion) {
        const body = {
            collection_id: collectionId,
            model_class: modelClass,
        };
        if (inputShape) body.input_shape = inputShape;
        if (opsetVersion) body.opset_version = opsetVersion;

        const resp = await fetch(`${this._base}/api/onnx/export`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const result = await resp.json();
        if (!resp.ok) throw new Error(result.error || `Export failed: ${resp.status}`);
        return result;
    }

    /**
     * Fetch the dataset samples assigned to a lease.
     *
     * @param {string} leaseId
     * @returns {Promise<object>}  { collection_id, slot_name, generation, samples }
     */
    async fetchDataset(leaseId) {
        const url = `${this._base}/api/web/lease/${encodeURIComponent(leaseId)}/dataset`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Dataset fetch failed: ${resp.status}`);
        return resp.json();
    }

    /**
     * Fetch a sample image as an HTMLImageElement (for canvas rendering).
     *
     * @param {string} slotName
     * @param {string} sampleId
     * @returns {Promise<HTMLImageElement>}
     */
    async fetchSampleImage(slotName, sampleId) {
        const url = `${this._base}/api/web/dataset/${encodeURIComponent(slotName)}/${encodeURIComponent(sampleId)}/image.png`;
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load image ${sampleId}`));
            img.src = url;
        });
    }

    /**
     * Extract weight deltas from the current training session and POST as
     * gradients (.npz) to the server.
     *
     * onnxruntime-web training builds expose getContiguousParameters() and
     * getParametersSize(); we snapshot before/after training and compute
     * the delta.  The delta is encoded in a minimal npz-compatible format.
     *
     * For inference-only sessions (no training API), this falls back to
     * extracting the output of a forward pass and sending it as a simple
     * JSON gradient blob that the server will wrap into an npz.
     *
     * @param {string}       leaseId          The lease to return gradients for
     * @param {Float32Array} originalWeights   Snapshot taken before training
     * @returns {Promise<object>}              Server response
     */
    async returnGradients(leaseId, originalWeights) {
        let deltaBytes;

        if (this._trainingSession) {
            // Extract current trained weights from ORT training session
            const currentSize = this._trainingSession.getParametersSize();
            const currentWeights = new Float32Array(currentSize);
            await this._trainingSession.getContiguousParameters(currentWeights);

            // Compute deltas
            const deltas = new Float32Array(currentWeights.length);
            for (let i = 0; i < deltas.length; i++) {
                deltas[i] = currentWeights[i] - (originalWeights[i] || 0);
            }
            deltaBytes = NodusOnnxClient._encodeNpz({"flat_delta": deltas});
        } else if (this._session && originalWeights) {
            // Inference-only fallback: caller must supply computed deltas directly
            deltaBytes = NodusOnnxClient._encodeNpz({"flat_delta": originalWeights});
        } else {
            throw new Error("No session available for gradient extraction");
        }

        const resp = await fetch(
            `${this._base}/api/lease/${encodeURIComponent(leaseId)}/gradients`,
            { method: "POST", body: deltaBytes }
        );
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`Gradient return failed: ${resp.status} ${err}`);
        }
        return resp.json();
    }

    /**
     * Snapshot the current training session weights (call BEFORE training).
     * @returns {Promise<Float32Array>}
     */
    async snapshotWeights() {
        if (!this._trainingSession) {
            throw new Error("Training session not initialized");
        }
        const size = this._trainingSession.getParametersSize();
        const buf = new Float32Array(size);
        await this._trainingSession.getContiguousParameters(buf);
        return buf;
    }

    /**
     * Submit a labeled image to the web dataset from the browser.
     *
     * @param {string}   slotName   Target vocabulary slot
     * @param {string}   imageB64   Base64-encoded PNG/JPEG
     * @param {string[]} labels     Label strings
     * @param {string}   [hint]     Optional client hint
     * @returns {Promise<object>}   Server response { sample_id, ... }
     */
    async submitSample(slotName, imageB64, labels, hint) {
        const body = { image_b64: imageB64, labels, client_hint: hint || "" };
        const resp = await fetch(
            `${this._base}/api/web/dataset/${encodeURIComponent(slotName)}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            }
        );
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`Submit failed: ${resp.status} ${err}`);
        }
        return resp.json();
    }

    /**
     * Fetch lease status for all slots.
     * @returns {Promise<object>}
     */
    async fetchLeaseStatus() {
        const resp = await fetch(`${this._base}/api/lease/status`);
        if (!resp.ok) throw new Error(`Status fetch failed: ${resp.status}`);
        return resp.json();
    }

    /**
     * Fetch collection detail.
     * @param {string} collectionId
     * @returns {Promise<object>}
     */
    async fetchCollection(collectionId) {
        const url = `${this._base}/api/lease/collection/${encodeURIComponent(collectionId)}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Collection fetch failed: ${resp.status}`);
        return resp.json();
    }

    /* ------------------------------------------------------------------ */
    /*  Cleanup                                                           */
    /* ------------------------------------------------------------------ */

    async destroy() {
        if (this._trainingSession) {
            await this._trainingSession.release();
            this._trainingSession = null;
        }
        if (this._session) {
            await this._session.release();
            this._session = null;
        }
    }

    /* ------------------------------------------------------------------ */
    /*  NPZ encoder (minimal — single array per key, float32 only)       */
    /* ------------------------------------------------------------------ */

    /**
     * Encode a dict of {name: Float32Array} into a numpy .npz byte stream.
     * The .npz format is a ZIP archive of .npy files.
     *
     * @param {Object<string, Float32Array>} arrays
     * @returns {Uint8Array}
     */
    static _encodeNpz(arrays) {
        const enc = new TextEncoder();
        const files = [];
        for (const [name, arr] of Object.entries(arrays)) {
            const npy = NodusOnnxClient._encodeNpy(arr);
            files.push({ name: name + ".npy", data: npy });
        }
        return NodusOnnxClient._buildZip(files);
    }

    /**
     * Encode a Float32Array as a .npy file (numpy format v1.0).
     * @param {Float32Array} arr
     * @returns {Uint8Array}
     */
    static _encodeNpy(arr) {
        // NPY header: magic "\x93NUMPY", version 1.0, header_len (LE uint16), then header string
        const shape = `(${arr.length},)`;
        const header = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shape}, }`;
        // Pad header to 64-byte alignment (magic=6 + version=2 + header_len=2 + header + \n)
        const preambleLen = 10; // 6 + 2 + 2
        let padded = header;
        while ((preambleLen + padded.length + 1) % 64 !== 0) padded += " ";
        padded += "\n";

        const headerBytes = enc(padded);
        const buf = new Uint8Array(preambleLen + headerBytes.length + arr.byteLength);
        buf[0] = 0x93; buf[1] = 0x4e; buf[2] = 0x55; buf[3] = 0x4d;
        buf[4] = 0x50; buf[5] = 0x59; // \x93NUMPY
        buf[6] = 1; buf[7] = 0; // version 1.0
        const hl = headerBytes.length;
        buf[8] = hl & 0xff; buf[9] = (hl >> 8) & 0xff; // header_len LE uint16
        buf.set(headerBytes, 10);
        buf.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength), preambleLen + headerBytes.length);
        return buf;

        function enc(s) { return new TextEncoder().encode(s); }
    }

    /**
     * Build a ZIP archive from an array of {name, data: Uint8Array}.
     * No compression (STORE method) — keeps the browser-side code tiny.
     * @param {{name: string, data: Uint8Array}[]} files
     * @returns {Uint8Array}
     */
    static _buildZip(files) {
        const enc = new TextEncoder();
        const parts = [];         // local file header + data
        const centralDir = [];    // central directory entries
        let offset = 0;

        for (const f of files) {
            const nameBytes = enc.encode(f.name);
            // Local file header (30 + name_length)
            const lfh = new Uint8Array(30 + nameBytes.length);
            const v = new DataView(lfh.buffer);
            v.setUint32(0, 0x04034b50, true);  // local file header signature
            v.setUint16(4, 20, true);           // version needed
            v.setUint16(6, 0, true);            // flags
            v.setUint16(8, 0, true);            // compression: STORE
            v.setUint16(10, 0, true);           // mod time
            v.setUint16(12, 0, true);           // mod date
            v.setUint32(14, NodusOnnxClient._crc32(f.data), true);
            v.setUint32(18, f.data.length, true);   // compressed size
            v.setUint32(22, f.data.length, true);   // uncompressed size
            v.setUint16(26, nameBytes.length, true); // filename length
            v.setUint16(28, 0, true);                // extra field length
            lfh.set(nameBytes, 30);

            // Central directory entry (46 + name_length)
            const cd = new Uint8Array(46 + nameBytes.length);
            const cv = new DataView(cd.buffer);
            cv.setUint32(0, 0x02014b50, true);  // central dir signature
            cv.setUint16(4, 20, true);           // version made by
            cv.setUint16(6, 20, true);           // version needed
            cv.setUint16(8, 0, true);            // flags
            cv.setUint16(10, 0, true);           // compression: STORE
            cv.setUint16(12, 0, true);           // mod time
            cv.setUint16(14, 0, true);           // mod date
            cv.setUint32(16, NodusOnnxClient._crc32(f.data), true);
            cv.setUint32(20, f.data.length, true);   // compressed
            cv.setUint32(24, f.data.length, true);   // uncompressed
            cv.setUint16(28, nameBytes.length, true); // filename length
            cv.setUint16(30, 0, true);                // extra field length
            cv.setUint16(32, 0, true);                // comment length
            cv.setUint16(34, 0, true);                // disk number
            cv.setUint16(36, 0, true);                // internal attrs
            cv.setUint32(38, 0, true);                // external attrs
            cv.setUint32(42, offset, true);            // local header offset
            cd.set(nameBytes, 46);

            parts.push(lfh, f.data);
            centralDir.push(cd);
            offset += lfh.length + f.data.length;
        }

        // End of central directory record
        const cdOffset = offset;
        let cdSize = 0;
        for (const cd of centralDir) cdSize += cd.length;

        const eocd = new Uint8Array(22);
        const ev = new DataView(eocd.buffer);
        ev.setUint32(0, 0x06054b50, true);
        ev.setUint16(4, 0, true);  // disk number
        ev.setUint16(6, 0, true);  // disk of central dir
        ev.setUint16(8, files.length, true);
        ev.setUint16(10, files.length, true);
        ev.setUint32(12, cdSize, true);
        ev.setUint32(16, cdOffset, true);
        ev.setUint16(20, 0, true); // comment length

        const totalLen = offset + cdSize + 22;
        const out = new Uint8Array(totalLen);
        let pos = 0;
        for (const p of parts) { out.set(p, pos); pos += p.length; }
        for (const c of centralDir) { out.set(c, pos); pos += c.length; }
        out.set(eocd, pos);
        return out;
    }

    /** CRC32 (ISO 3309) for Uint8Array. */
    static _crc32(data) {
        let crc = 0xFFFFFFFF;
        if (!NodusOnnxClient._crcTable) {
            const t = new Uint32Array(256);
            for (let n = 0; n < 256; n++) {
                let c = n;
                for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
                t[n] = c;
            }
            NodusOnnxClient._crcTable = t;
        }
        const t = NodusOnnxClient._crcTable;
        for (let i = 0; i < data.length; i++) {
            crc = t[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8);
        }
        return (crc ^ 0xFFFFFFFF) >>> 0;
    }
}
NodusOnnxClient._crcTable = null;
