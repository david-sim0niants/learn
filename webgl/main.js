const canvas = document.getElementById("main-canvas");
/** @type {WebGL2RenderingContext} */
const gl = canvas.getContext("webgl2");

const triangleVertices = new Float32Array(
    [
        [0.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
    ].flat(),
);

const triangleColors = new Float32Array(
    [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ].flat(),
);

const compileShader = (id, shader_type) => {
    let shaderElement = document.getElementById(id);
    let shaderSource = shaderElement.text.trim();

    let shader = gl.createShader(shader_type);

    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(shader));
    return shader;
};

const vertexShader = compileShader("vertex-shader", gl.VERTEX_SHADER);
const fragmentShader = compileShader("fragment-shader", gl.FRAGMENT_SHADER);

const shaderProgram = gl.createProgram();
gl.attachShader(shaderProgram, vertexShader);
gl.attachShader(shaderProgram, fragmentShader);
gl.linkProgram(shaderProgram);

if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS))
    throw new Error("Could not link shaders");

gl.useProgram(shaderProgram);

// Vertices setup
const triangleVertexPositionBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);

const positionAttributeLocation = gl.getAttribLocation(
    shaderProgram,
    "position",
);
gl.enableVertexAttribArray(positionAttributeLocation);
gl.vertexAttribPointer(positionAttributeLocation, 2, gl.FLOAT, false, 0, 0);

// Colors setup
const triangleVertexColorBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexColorBuffer);
gl.bufferData(gl.ARRAY_BUFFER, triangleColors, gl.STATIC_DRAW);

const colorAttributeLocation = gl.getAttribLocation(shaderProgram, "color");

gl.enableVertexAttribArray(colorAttributeLocation);
gl.vertexAttribPointer(colorAttributeLocation, 4, gl.FLOAT, false, 0, 0);

const runRenderLoop = () => {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    requestAnimationFrame(runRenderLoop);
};

requestAnimationFrame(runRenderLoop);

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener("resize", resize);
resize();
