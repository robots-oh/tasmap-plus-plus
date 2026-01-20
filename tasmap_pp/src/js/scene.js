import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from     'three/addons/loaders/OBJLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { GUI } from           'three/addons/libs/lil-gui.module.min.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

let num_objects_curr = 0;
let num_objects = 100;



function add_progress_bar(){
    let gProgressElement = document.createElement("div");
    const html_code = '<div class="progress">\n' +
		'<div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%" id="progress_bar"></div>\n' +
		'</div>';
    gProgressElement.innerHTML = html_code;
    gProgressElement.id = "progress_bar_id"
    gProgressElement.style.left = "20%";
    gProgressElement.style.right = "20%";
    gProgressElement.style.position = "fixed";
    gProgressElement.style.top = "50%";
    document.body.appendChild(gProgressElement);
}

function step_progress_bar(){
	num_objects_curr += 1.0
	let progress_int = parseInt(num_objects_curr / num_objects * 100.0)
	let width_string = String(progress_int)+'%';
	document.getElementById('progress_bar').style.width = width_string;
	document.getElementById('progress_bar').innerText = width_string;

	if (progress_int==100) {
		document.getElementById( 'progress_bar_id' ).innerHTML = "";
	}
}


function set_camera_properties(properties){
	camera.setFocalLength(properties['focal_length']);
	console.log(camera.getFocalLength);
	camera.up.set(properties['up'][0],
		          properties['up'][1],
				  properties['up'][2]);
	camera.position.set(properties['position'][0],
						properties['position'][1],
						properties['position'][2]);
	update_controls();
	controls.update();
	controls.target = new THREE.Vector3(properties['look_at'][0],
	 	                                properties['look_at'][1],
	 						    		properties['look_at'][2]);
	camera.updateProjectionMatrix();
	controls.update();
}



function halfToFloat(value) {        // 16-bit → 32-bit
	return THREE.DataUtils.fromHalfFloat(value);  // r152+ 내장
  }
 
function get_points(properties){
	const n   = properties.num_points;
	const url = properties.binary_filename;
  
	const geom      = new THREE.BufferGeometry();
	const pos32     = new Float32Array(n*3);
	const col32     = new Float32Array(n*3);

	fetch(url).then(r => r.arrayBuffer()).then(buf => {
		const dv         = new DataView(buf);
		const BYTES_POS  = 6 * n;
		const BYTES_COL  = 3 * n;

		for (let i=0; i<n; i++) {
			const pOff = i * 6;
			const cOff = BYTES_POS + i * 3;

			// XYZ (float16 → float32)
			pos32[i*3]     = halfToFloat(dv.getUint16(pOff    , true)) * 0.001;
			pos32[i*3 + 1] = halfToFloat(dv.getUint16(pOff + 2, true)) * 0.001;
			pos32[i*3 + 2] = halfToFloat(dv.getUint16(pOff + 4, true)) * 0.001;

			// RGB (uint8 → float32)
			col32[i*3]     = dv.getUint8(cOff    ) / 255;
			col32[i*3 + 1] = dv.getUint8(cOff + 1) / 255;
			col32[i*3 + 2] = dv.getUint8(cOff + 2) / 255;
		}

		geom.setAttribute('position', new THREE.Float32BufferAttribute(pos32, 3));
		geom.setAttribute('color',    new THREE.Float32BufferAttribute(col32, 3));
		
	})
	.then(step_progress_bar)
	.then(render);

	let uniforms = {
		pointSize: { value: properties['point_size'] },
		alpha: { value: properties['alpha'] },
		shading_type: { value: 0 },  
	};

	let material = new THREE.ShaderMaterial({
		uniforms: uniforms,
		vertexShader: document.getElementById('vertexshader').textContent,
		fragmentShader: document.getElementById('fragmentshader').textContent,
		transparent: true,
	});

	let points = new THREE.Points(geom, material);
	return points;
}



const task_colors = {
	'leave':      'rgb(199, 196, 194)',    
	'relocate':   'rgb(98, 194, 95)',   
	'reorient':   'rgb(238, 216, 89)',  
	'washing-up': 'rgb(95, 235, 216)',  
	'mop':        'rgb(1, 58, 192)',    
	'vacuum':     'rgb(189, 99, 25)',   
	'wipe':       'rgb(79, 147, 173)',  
	'fold':       'rgb(61, 122, 66)',   
	'close':      'rgb(122, 93, 59)',   
	'turn-off':   'rgb(250, 129, 73)',  
	'dispose':    'rgb(142, 64, 179)',  
	'empty':      'rgb(173, 67, 108)',  
};

function get_task_color(labelText) {
	const match = labelText.match(/\[(.*?)\]/);  
	if (match && match[1]) {
		const tasks = match[1].split(',');
		const task = tasks[0].trim()	

		const task_color = task_colors[task]
		return task_color; 
	}
	return '';  
}



function makeLabelForBbox(bboxProps) {
    const div = document.createElement('div');
    div.className = 'label';

    const text = `${bboxProps.label}\n${bboxProps.task}`;
    div.innerText = text;
	div.style.textAlign = 'center';

    const taskColor = get_task_color(text);       
    const isDark   = (c) => {
        const [r,g,b] = c.match(/\d+/g).map(Number);
        return (0.299*r + 0.587*g + 0.114*b) < 110;
    };
    div.style.padding         = '1px 4px';
    div.style.fontSize        = '12px';
    div.style.borderRadius    = '2px';
	div.style.fontFamily = 'Roboto, sans-serif';
	div.style.fontWeight = '400';
    div.style.backgroundColor = taskColor;
    div.style.color           = isDark(taskColor) ? 'white' : 'black';
	div.style.webkitTextStroke = '0';      
	div.style.textShadow = 'none';  
	div.style.padding = '1px 4px'; 

    const label2d = new CSS2DObject(div);
    label2d.position.set(
        bboxProps.position[0],
        bboxProps.position[1],
        bboxProps.position[2]
    );
    return label2d;   
}


function get_material(alpha){
	let uniforms = {
		alpha: {value: alpha},
		shading_type: {value: 1},
	};
	let material = new THREE.ShaderMaterial({
		uniforms:       uniforms,
		vertexShader:   document.getElementById('vertexshader').textContent,
		fragmentShader: document.getElementById('fragmentshader').textContent,
		transparent:    true,
    });
    return material;
}

function set_geometry_vertex_color(geometry, color){
	const r = Math.fround(color[0] / 255.0);
	const g = Math.fround(color[1] / 255.0);
	const b = Math.fround(color[2] / 255.0);
	const num_vertices = geometry.getAttribute('position').count;
	const colors = new Float32Array(num_vertices * 3);
	for (let i = 0; i < num_vertices; i++){
		colors[3 * i + 0] = r;
		colors[3 * i + 1] = g;
		colors[3 * i + 2] = b;
	}
	geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

function get_cylinder_geometry(radius_top, radius_bottom, height, radial_segments, color){
	let geometry = new THREE.CylinderGeometry(radius_top, radius_bottom, height, radial_segments);
	set_geometry_vertex_color(geometry, color)
	return geometry;
}

function get_sphere_geometry(radius, widthSegments, heightSegments, color){
	const geometry = new THREE.SphereGeometry(radius, widthSegments, heightSegments);
	set_geometry_vertex_color(geometry, color);
	return geometry;
}

function get_cuboid(properties){
	const radius_top = properties['edge_width'];
	const radius_bottom = properties['edge_width'];
	const radial_segments = 30;
	const height = 1;
	
	let geometry = get_cylinder_geometry(
		radius_top, radius_bottom, height, radial_segments,
		properties['color']);
	let material = get_material(properties['alpha']);

	const cylinder_x = new THREE.Mesh(geometry, material);
	cylinder_x.scale.set(1.0, properties['size'][0], 1.0)
	cylinder_x.rotateZ(3.1415/2.0)
	const cylinder_00 = cylinder_x.clone()
	cylinder_00.position.set(0, -properties['size'][1]/2.0, -properties['size'][2]/2.0)
	const cylinder_01 = cylinder_x.clone()
	cylinder_01.position.set(0, properties['size'][1]/2.0, -properties['size'][2]/2.0)
	const cylinder_20 = cylinder_x.clone()
	cylinder_20.position.set(0, -properties['size'][1]/2.0, properties['size'][2]/2.0)
	const cylinder_21 = cylinder_x.clone()
	cylinder_21.position.set(0, properties['size'][1]/2.0, properties['size'][2]/2.0)

	const cylinder_y = new THREE.Mesh(geometry, material);
	cylinder_y.scale.set(1.0, properties['size'][1], 1.0)
	const cylinder_02 = cylinder_y.clone()
	cylinder_02.position.set(-properties['size'][0]/2.0, 0, -properties['size'][2]/2.0)
	const cylinder_03 = cylinder_y.clone()
	cylinder_03.position.set(properties['size'][0]/2.0, 0, -properties['size'][2]/2.0)
	const cylinder_22 = cylinder_y.clone()
	cylinder_22.position.set(-properties['size'][0]/2.0, 0, properties['size'][2]/2.0)
	const cylinder_23 = cylinder_y.clone()
	cylinder_23.position.set(properties['size'][0]/2.0, 0, properties['size'][2]/2.0)

	const cylinder_z = new THREE.Mesh(geometry, material);
	cylinder_z.scale.set(1.0, properties['size'][2], 1.0)
	cylinder_z.rotateX(3.1415/2.0)
	const cylinder_10 = cylinder_z.clone()
	cylinder_10.position.set(-properties['size'][0]/2.0, -properties['size'][1]/2.0, 0.0)
	const cylinder_11 = cylinder_z.clone()
	cylinder_11.position.set(properties['size'][0]/2.0, -properties['size'][1]/2.0, 0.0)
	const cylinder_12 = cylinder_z.clone()
	cylinder_12.position.set(-properties['size'][0]/2.0, properties['size'][1]/2.0, 0.0)
	const cylinder_13 = cylinder_z.clone()
	cylinder_13.position.set(properties['size'][0]/2.0, properties['size'][1]/2.0, 0.0)

	let corner_geometry = get_sphere_geometry(properties['edge_width'], 30, 30, properties['color']);

	const sphere = new THREE.Mesh(corner_geometry, material);
	const corner_00 = sphere.clone()
	corner_00.position.set(-properties['size'][0]/2.0, -properties['size'][1]/2.0, -properties['size'][2]/2.0)
	const corner_01 = sphere.clone()
	corner_01.position.set(properties['size'][0]/2.0, -properties['size'][1]/2.0, -properties['size'][2]/2.0)
	const corner_02 = sphere.clone()
	corner_02.position.set(-properties['size'][0]/2.0, properties['size'][1]/2.0, -properties['size'][2]/2.0)
	const corner_03 = sphere.clone()
	corner_03.position.set(properties['size'][0]/2.0, properties['size'][1]/2.0, -properties['size'][2]/2.0)
	const corner_10 = sphere.clone()
	corner_10.position.set(-properties['size'][0]/2.0, -properties['size'][1]/2.0, properties['size'][2]/2.0)
	const corner_11 = sphere.clone()
	corner_11.position.set(properties['size'][0]/2.0, -properties['size'][1]/2.0, properties['size'][2]/2.0)
	const corner_12 = sphere.clone()
	corner_12.position.set(-properties['size'][0]/2.0, properties['size'][1]/2.0, properties['size'][2]/2.0)
	const corner_13 = sphere.clone()
	corner_13.position.set(properties['size'][0]/2.0, properties['size'][1]/2.0, properties['size'][2]/2.0)

	const cuboid = new THREE.Group();
	cuboid.position.set(properties['position'][0], properties['position'][1], properties['position'][2])
	cuboid.add(cylinder_00)
	cuboid.add(cylinder_01)
	cuboid.add(cylinder_20)
	cuboid.add(cylinder_21)
	cuboid.add(cylinder_02)
	cuboid.add(cylinder_03)
	cuboid.add(cylinder_22)
	cuboid.add(cylinder_23)
	cuboid.add(cylinder_10)
	cuboid.add(cylinder_11)
	cuboid.add(cylinder_12)
	cuboid.add(cylinder_13)

	cuboid.add(corner_00)
	cuboid.add(corner_01)
	cuboid.add(corner_02)
	cuboid.add(corner_03)
	cuboid.add(corner_10)
	cuboid.add(corner_11)
	cuboid.add(corner_12)
	cuboid.add(corner_13)

	const q = new THREE.Quaternion(
			properties['orientation'][0],
			properties['orientation'][1],
			properties['orientation'][2],
			properties['orientation'][3])
	cuboid.setRotationFromQuaternion(q)
	cuboid.position.set(properties['position'][0], properties['position'][1], properties['position'][2])
	return cuboid
}



function init_gui(objects) {

    const menuMap          = new Map();   
    const groupMembers     = new Map();  
    const groupControllers = new Map();   

    for (const [name, obj] of Object.entries(objects)) {
        const splits = name.split(';');          

        if (splits.length > 1) {
            const folderName = splits[0];        
            const itemName   = splits[1];        

            if (!menuMap.has(folderName)) {
                const folder = gui.addFolder(folderName);
                menuMap.set(folderName, folder);
                groupMembers.set(folderName, []);
                groupControllers.set(folderName, []);
            }

            const folder = menuMap.get(folderName);
            groupMembers.get(folderName).push(obj);

            const ctrl = folder.add(obj, 'visible').name(itemName).onChange((v) => {
                obj.visible = v;                              

                obj.children?.forEach(child => {
                    if (child.isObject3D) child.visible = v;  
                    if (child.element)                       
                        child.element.style.display = v ? 'block' : 'none';
                });

                render();
            });

            groupControllers.get(folderName).push(ctrl);
            folder.open();
        }

        else {
            gui.add(obj, 'visible').name(name).onChange(render);
        }
    }

    for (const [folderName, members] of groupMembers.entries()) {
        const folder      = menuMap.get(folderName);
        const controllers = groupControllers.get(folderName);

        const wrapper = { visible: true };
        folder.add(wrapper, 'visible').name('[Toggle All]').onChange((v) => {
            members.forEach((o, i) => {
                o.visible = v;
                controllers[i].setValue(v);                 

                o.children?.forEach(child => {
                    if (child.isObject3D) child.visible = v;
                    if (child.element)
                        child.element.style.display = v ? 'block' : 'none';
                });
            });
            render();
        });
    }
}


function createTaskLegend(task_colors) {
	const legendDiv = document.getElementById('task-legend');
	legendDiv.innerHTML = ''; 

	for (const [task, color] of Object.entries(task_colors)) {
		const row = document.createElement('div');
		row.style.display = 'flex';
		row.style.alignItems = 'center';
		row.style.marginBottom = '0px';

		const colorBox = document.createElement('div');
		colorBox.style.width = '20px';
		colorBox.style.height = '20px';
		colorBox.style.borderRadius = '3px';
		colorBox.style.marginRight = '6px';
		colorBox.style.backgroundColor = color;

		const label = document.createElement('span');
		label.textContent = task;

		row.appendChild(colorBox);
		row.appendChild(label);
		legendDiv.appendChild(row);
	}
}


function render() {
    renderer.render(scene, camera);
	labelRenderer.render(scene, camera);
}

function init(){
	scene.background = new THREE.Color(0xffffff);
	renderer.setSize(window.innerWidth, window.innerHeight);
	labelRenderer.setSize(window.innerWidth, window.innerHeight);

	let hemiLight = new THREE.HemisphereLight( 0xffffff, 0x444444 );
	hemiLight.position.set(0, 20, 0);
	//scene.add(hemiLight);

	let dirLight = new THREE.DirectionalLight( 0xffffff );
	dirLight.position.set(-10, 10, - 10);
	dirLight.castShadow = true;
	dirLight.shadow.camera.top = 2;
	dirLight.shadow.camera.bottom = - 2;
	dirLight.shadow.camera.left = - 2;
	dirLight.shadow.camera.right = 2;
	dirLight.shadow.camera.near = 0.1;
	dirLight.shadow.camera.far = 40;
	//scene.add(dirLight);

	let intensity = 0.5;
	let color = 0xffffff;
	const spotLight1 = new THREE.SpotLight(color, intensity);
	spotLight1.position.set(100, 1000, 0);
	scene.add(spotLight1);
	const spotLight2 = new THREE.SpotLight(color, intensity/3.0);
	spotLight2.position.set(100, -1000, 0);
	scene.add(spotLight2);
	const spotLight3 = new THREE.SpotLight(color, intensity);
	spotLight3.position.set(0, 100, 1000);
	scene.add(spotLight3);
	const spotLight4 = new THREE.SpotLight(color, intensity/3.0);
	spotLight4.position.set(0, 100, -1000);
	scene.add(spotLight4);
	const spotLight5 = new THREE.SpotLight(color, intensity);
	spotLight5.position.set(1000, 0, 100);
	scene.add(spotLight5);
	const spotLight6 = new THREE.SpotLight(color, intensity/3.0);
	spotLight6.position.set(-1000, 0, 100);
	scene.add(spotLight6);

	raycaster = new THREE.Raycaster();
	raycaster.params.Points.threshold = 1.0;
}

function create_threejs_objects(properties){

	num_objects_curr = 0.0;
	num_objects = parseFloat(Object.entries(properties).length);

	for (const [object_name, object_properties] of Object.entries(properties)) {
		if (String(object_properties['type']).localeCompare('camera') == 0){
			set_camera_properties(object_properties);
			render();
    		step_progress_bar();
    		continue;
		}
		if (String(object_properties['type']).localeCompare('points') == 0){
			threejs_objects[object_name] = get_points(object_properties);
    		render();
		}
		if (String(object_properties['type']).localeCompare('cuboid') == 0){
			const cuboid = get_cuboid(object_properties);
			if (object_properties['label_visible']){
				const bboxLabel = makeLabelForBbox(object_properties);
				bboxLabel.position.set(0, 0, 0);
				cuboid.add(bboxLabel);}
			threejs_objects[object_name] = cuboid;
			step_progress_bar();
			render();
		}
		threejs_objects[object_name].visible = object_properties['visible'];
		threejs_objects[object_name].frustumCulled = false;
	}
	
	// // Add axis helper
	// threejs_objects['Axis'] = new THREE.AxesHelper(1);

	render();
}

function add_threejs_objects_to_scene(threejs_objects){
	for (const [key, value] of Object.entries(threejs_objects)) {
		scene.add(value);
	}
}

function onWindowResize(){
    const innerWidth = window.innerWidth
    const innerHeight = window.innerHeight;
    renderer.setSize(innerWidth, innerHeight);
    labelRenderer.setSize(innerWidth, innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    render();
}

function update_controls(){
	controls = new OrbitControls(camera, labelRenderer.domElement);
	controls.addEventListener("change", render);
	controls.enableKeys = true;
	controls.enablePan = true; // enable dragging
}

const scene = new THREE.Scene();

const renderer = new THREE.WebGLRenderer({antialias: true});
document.getElementById('render_container').appendChild(renderer.domElement)

var camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.01, 1000);
var controls = '';

let labelRenderer = new CSS2DRenderer();
labelRenderer.setSize( window.innerWidth, window.innerHeight );
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0px';
document.getElementById('render_container').appendChild(labelRenderer.domElement)

window.addEventListener('resize', onWindowResize, false);

let raycaster;
let intersection = null;
let mouse = new THREE.Vector2();

const gui = new GUI({autoPlace: true, width: 230});

let threejs_objects = {};

init();

fetch('nodes.json')
	.then(response => {add_progress_bar(); return response;})
    .then(response => {return response.json();})
    // .then(json_response => {console.log(json_response); return json_response})
    .then(json_response => create_threejs_objects(json_response))
    .then(() => add_threejs_objects_to_scene(threejs_objects))
    .then(() => init_gui(threejs_objects))
	.then(() => createTaskLegend(task_colors))
	.then(() => console.log('Done'))
	.then(render);

