
const tf = require('@tensorflow/tfjs-node');

const {encoder, decoder, vae, vaeLoss} = require('./model');

function main() {
        const opts = {
            originalDim: 100,
            intermediateDim: 10,
            latentDim: 2
        };
        const enc = encoder(opts);
        const dec = decoder(opts);
        const model = vae(enc, dec);
        expect(model.inputs.length).toEqual(1);
        expect(model.inputs[0].shape).toEqual([null, 100]);
        expect(model.outputs.length).toEqual(4);
        expect(model.outputs[0].shape).toEqual([null, 100]);
        expect(model.outputs[1].shape).toEqual([null, 2]);
        expect(model.outputs[2].shape).toEqual([null, 2]);
        expect(model.outputs[3].shape).toEqual([null, 2]);

        const numExamples = 4;
        const xs = tf.randomUniform([numExamples, 100]);
        const ys = model.predict(xs);
        expect(ys.length).toEqual(4);
        expect(ys[0].shape).toEqual([numExamples, 100]);
        expect(ys[1].shape).toEqual([numExamples, 2]);
        expect(ys[2].shape).toEqual([numExamples, 2]);
        expect(ys[3].shape).toEqual([numExamples, 2]);

        const numTensors0 = tf.memory().numTensors;
        const loss = vaeLoss(xs, ys);
        const numTensors1 = tf.memory().numTensors;
        expect(numTensors1).toEqual(numTensors0 + 1);
        expect(loss.shape).toEqual([]);  // loss is a scalar.
        expect(loss.arraySync()).toBeGreaterThan(0);
}
