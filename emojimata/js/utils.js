const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const parseConsts = model_graph => {
    const dtypes = {
        'DT_INT32': ['int32', 'intVal', Int32Array],
        'DT_FLOAT': ['float32', 'floatVal', Float32Array]
    };
    
    const consts = {};
    model_graph.modelTopology.node.filter(n => n.op === 'Const').forEach((node) => {
        const v = node.attr.value.tensor;
        if (!v || !v.dtype || !dtypes[v.dtype]) {
            console.warn(`Invalid tensor for node ${node.name}`);
            return;
        }
        
        const [dtype, field, arrayType] = dtypes[v.dtype];
        
        try {
            if (!v.tensorShape.dim) {
                if (v[field] && v[field].length > 0) {
                    consts[node.name] = [tf.scalar(v[field][0], dtype)];
                } else {
                    consts[node.name] = [tf.scalar(0, dtype)];
                }
            } else {
                // if there is a 0-length dimension, the exported graph json lacks "size"
                const shape = v.tensorShape.dim.map(d => (!d.size) ? 0 : parseInt(d.size));
                let arr;
                if (v.tensorContent) {
                    const data = atob(v.tensorContent);
                    const buf = new Uint8Array(data.length);
                    for (var i = 0; i < data.length; ++i) {
                        buf[i] = data.charCodeAt(i);
                    }
                    arr = new arrayType(buf.buffer);
                } else {
                    const size = shape.reduce((a, b) => a * b);
                    arr = new arrayType(size);
                    if (size && v[field] && v[field].length > 0) {
                        arr.fill(v[field][0]);
                    }
                }
                consts[node.name] = [tf.tensor(arr, shape, dtype)];
            }
        } catch (error) {
            console.error(`Error creating tensor for node ${node.name}:`, error);
            // Provide a safe fallback
            consts[node.name] = [tf.scalar(0, dtype)];
        }
    });
    return consts;
}; 