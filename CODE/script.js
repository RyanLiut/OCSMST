// 全局变量
let networks = []; // 动态数量的网络图实例
let nodeDatasets = []; // 动态数量的独立节点数据集
let edgeDatasets = []; // 动态数量的独立边数据集
let nodeIdCounters = []; // 每个图的节点ID计数器
let selectedNodes = [];
let selectedEdges = [];
let currentFormsData = []; // 存储当前的forms数据
let currentFormGroups = {}; // 存储当前的语组分组数据
let highlightedNodes = []; // 存储当前高亮的节点
let currentFormOutlines = []; // [{formName, color, components: [[nodeId,...],...]}]
let originalExcelData = []; // 存储原始Excel数据
let supplementaryFeatureLabels = {}; // 缩写/中文名 -> {chinese, english, abbrev}
let supplementaryFeatureExamples = {}; // "form|feature" -> example text
let currentSelectedFormData = null; // 当前filter选中的form数据

const FORM_COLORS = [
    '#e53935','#fb8c00','#43a047','#8e24aa','#f4511e',
    '#7cb342','#c43e00','#d81b60','#9e9d24','#6d4c41',
    '#00897b','#5e35b1','#ff6d00','#00c853','#e65100',
    '#ad1457','#1e88e5','#546e7a','#039be5','#283593'
];
let currentLanguageHighlight = null;

let outlineDragging = null;
let outlineHoverVertex = null;
let outlineUserOffsets = {};

let isBeautified = false; // 跟踪图形美化状态

// Evolution mode state
let isEvolutionMode = false;
let evolutionData = null;
let evolutionFeatureToNodeId = {};
let selectedEvolutionWords = [];
let selectedEvolutionDynasty = 0;

// SLA mode state
let isSLAMode = false;
let slaSheet1Languages = [];
let slaSheet1Forms = [];
let slaSavedGraphData = null;
let currentExampleName = '';
let llmSimCsvText = null; // stores uploaded LLM similarity matrix CSV text
let gtMatrix = null; // stores GT adjacency matrix (dedup-space) for edge comparison
let gtSlideIndex = -1; // index of the GT slide in the carousel (-1 if none)
let perExampleLabels = {}; // per-example feature label mapping from Sheet3
let labelLanguage = localStorage.getItem('labelLanguage') || 'zh'; // 'zh' or 'en'

// SLA proficiency level outlines (separate from form outlines)
let slaLevelOutlines = []; // [{ levelName, color, nodeIds, isActive }]
let slaSheet5LevelFeatures = null; // { '初级': [...], '中级': [...], '高级': [...] }

let currentEditEdgeId = null;
let currentSlideIndex = 0; // 当前显示的轮播图索引
let totalSlides = 0; // 总幻灯片数量
let evaluationMetricsData = []; // 存储所有语义地图的评价指标数据

// Merge功能相关的状态管理
let isMergedStates = []; // 每个地图的merge状态数组
let backupEvaluationMetricsArray = []; // 每个地图备份merge前的评价指标
let backupUnconnectedFormsArray = []; // 每个地图备份merge前的未连接表单
let mergedEdgeIdsArray = []; // 每个地图存储merge的边ID数组

async function fetchSupplementaryMetadata() {
    try {
        const resp = await fetch('/api/supplementary-metadata');
        if (resp.ok) {
            const data = await resp.json();
            supplementaryFeatureLabels = data.feature_labels || {};
            supplementaryFeatureExamples = data.feature_examples || {};
        }
    } catch (e) {
        console.log('Supplementary metadata not available');
    }
}

fetchSupplementaryMetadata();

// 配置网络图选项
const options = {
    nodes: {
        shape: 'circle',
        font: {
            size: 16,
            face: 'Arial'
        },
        borderWidth: 2,
        shadow: false
    },
    edges: {
        shadow: false,
        font: {
            size: 12,
            align: 'middle',
            background: 'white',
            strokeWidth: 1,
            strokeColor: '#000000'
        },
        arrows: {
            to: { enabled: false }
        },
        color: {
            color: '#FFCCE5',       // 浅粉色
            highlight: '#FF69B4',   // 深粉色（热粉色）
            hover: '#FF69B4'        // 鼠标悬停时显示深粉色
        },
        chosen: {
            label: false
        },
        labelHighlightBold: false
    },
    physics: {
        enabled: true,
        barnesHut: {
            gravitationalConstant: -2000,
            centralGravity: 0.3,
            springLength: 150,
            springConstant: 0.04
        }
    },
    interaction: {
        hover: true,
        multiselect: true,
        navigationButtons: true,
        dragNodes: true,
        dragView: true,
        zoomView: true
    },
    manipulation: {
        enabled: false
    }
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化空的数据集数组
    nodeDatasets = [];
    edgeDatasets = [];
    networks = [];
    
    // 不再自动初始化网络图，等待用户上传文件后动态创建
    
    // 初始化筛选功能
    initializeFormFilter();
    
    // 初始化文件信息显示
    const fileInfo = document.getElementById('file-info');
    fileInfo.textContent = 'No file selected';
    
    // 绑定按钮事件
    document.getElementById('upload-btn').addEventListener('click', handleFileUpload);
    document.getElementById('download-guide-btn').addEventListener('click', handleDownloadGuide);
    document.getElementById('example-btn').addEventListener('click', handleExampleUpload);
document.getElementById('example-btn-2').addEventListener('click', handleExampleUpload2);
    document.getElementById('example-btn-3').addEventListener('click', handleExampleUpload3);
    document.getElementById('example-btn-4').addEventListener('click', handleExampleUpload4);
    document.getElementById('example-btn-5').addEventListener('click', handleExampleUpload5);
    document.getElementById('example-btn-6').addEventListener('click', handleExampleUpload6);
    document.getElementById('example-btn-7').addEventListener('click', handleExampleUpload7);
    document.getElementById('example-btn-8').addEventListener('click', handleExampleUpload8);
    document.getElementById('example-btn-9').addEventListener('click', handleExampleUpload9);
    document.getElementById('example-btn-10').addEventListener('click', handleExampleUpload10);
    document.getElementById('example-btn-11').addEventListener('click', handleExampleUpload11);
    document.getElementById('example-btn-12').addEventListener('click', handleExampleUpload12);
    document.getElementById('help-btn').addEventListener('click', showHelpModal);
    
    // Blend type selector
    const blendTypeBtns = document.querySelectorAll('.blend-type-btn');
    const blendSlider = document.getElementById('blend-weight-slider');
    const blendValueLabel = document.getElementById('blend-weight-value');
    const blendBestMarker = document.getElementById('blend-best-marker');
    
    blendTypeBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            blendTypeBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            const best = parseFloat(this.dataset.best);
            const step = parseFloat(this.dataset.step || '0.1');
            const max = parseFloat(this.dataset.max || '1');
            blendSlider.step = step;
            blendSlider.max = max;
            blendSlider.value = best;
            const decimals = step < 0.1 ? 2 : 1;
            blendValueLabel.textContent = best.toFixed(decimals);
            updateBestMarker(best);

            const isNoBlend = this.dataset.noblend === '1';
            const sliderWrapper = document.querySelector('.blend-slider-wrapper');
            sliderWrapper.style.display = isNoBlend ? 'none' : 'flex';

            const isEmbSim = this.dataset.calcType === 'L';
            const simUploadRow = document.getElementById('llm-sim-upload-row');
            const isEx2 = currentExampleName === 'supplementary adverbs.xlsx';
            if (isEmbSim && !isEx2) {
                simUploadRow.style.display = 'flex';
                if (!llmSimCsvText) {
                    alert('Please upload a function similarity matrix CSV file to use EmbSim blend.');
                    return;
                }
            } else {
                simUploadRow.style.display = 'none';
            }

            refetchGraph();
        });
    });
    
    blendSlider.addEventListener('input', function() {
        const val = parseFloat(this.value);
        const step = parseFloat(this.step || '0.1');
        const decimals = step < 0.1 ? 2 : 1;
        blendValueLabel.textContent = val.toFixed(decimals);
        updateBestMarker(val);
    });
    
    blendSlider.addEventListener('change', function() {
        refetchGraph();
    });
    
    function updateBestMarker(val) {
        const activeBtn = document.querySelector('.blend-type-btn.active');
        if (activeBtn) {
            const best = parseFloat(activeBtn.dataset.best);
            blendBestMarker.style.display = (Math.abs(val - best) < 0.001) ? 'inline' : 'none';
        }
    }
    
    // Re-fetch graph with current blend settings (if data is already loaded)
    async function refetchGraph() {
        if (!originalExcelData || !originalExcelData.data) return;
        const fileInfo = document.getElementById('file-info');
        fileInfo.textContent = 'Updating graph with new blend settings...';
        try {
            const response = await sendDataToBackend(originalExcelData);
            const fileName = currentExampleName || (document.getElementById('excel-file').files[0]?.name || 'Graph');
            if (response.graph_data && response.forms_with_nodes) {
                const graphDataArray = response.graph_data;
                const formsWithNodes = response.forms_with_nodes;
                window.currentFormsData = formsWithNodes;
                window.currentFormGroups = response.form_groups || {};
                updateGraph(graphDataArray);
                updateGraphTitles(fileName, graphDataArray);
                updateFormFilter(formsWithNodes, response.form_groups || {});
            } else if (Array.isArray(response)) {
                updateGraph(response);
                updateGraphTitles(fileName, response);
                window.currentFormsData = [];
                window.currentFormGroups = {};
                updateFormFilter([], {});
            } else {
                updateGraph([response]);
                updateGraphTitles(fileName, [response]);
                window.currentFormsData = [];
                window.currentFormGroups = {};
                updateFormFilter([], {});
            }
            fileInfo.textContent = `Loaded file: ${fileName}`;
        } catch (error) {
            console.error('Blend re-fetch failed:', error);
            fileInfo.textContent = `Update failed: ${error.message}`;
        }
    }
    
    // Initialize marker for default selection
    updateBestMarker(parseFloat(blendSlider.value));
    
    // LLM similarity matrix file upload
    document.getElementById('llm-sim-select-btn').addEventListener('click', function() {
        document.getElementById('llm-sim-file').click();
    });
    document.getElementById('llm-sim-file').addEventListener('change', function() {
        const fileInfo = document.getElementById('llm-sim-file-info');
        if (this.files && this.files.length > 0) {
            const file = this.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                llmSimCsvText = e.target.result;
                fileInfo.textContent = file.name;
                fileInfo.style.color = '#27ae60';
                refetchGraph();
            };
            reader.onerror = function() {
                fileInfo.textContent = 'Read failed';
                fileInfo.style.color = '#e74c3c';
            };
            reader.readAsText(file);
        }
    });
    
    // 绑定自定义文件选择按钮事件
    document.getElementById('file-select-btn').addEventListener('click', function() {
        document.getElementById('excel-file').click();
    });
    
    // 绑定文件输入change事件
    document.getElementById('excel-file').addEventListener('change', function() {
        const fileInfo = document.getElementById('file-info');
        const fileSelectBtn = document.getElementById('file-select-btn');
        if (this.files && this.files.length > 0) {
            const fileName = this.files[0].name;
            fileInfo.textContent = `Selected file: ${fileName}`;
            fileSelectBtn.textContent = fileName;
        } else {
             fileInfo.textContent = 'No file selected';
             fileSelectBtn.textContent = 'Choose File';
         }
     });
    document.getElementById('add-edge-btn').addEventListener('click', showAddEdgeForm);
    document.getElementById('edit-edge-btn').addEventListener('click', showEditEdgeForm);
    document.getElementById('delete-edge-btn').addEventListener('click', deleteSelectedEdge);
    document.getElementById('merge-edge-btn').addEventListener('click', function() {
        const currentIndex = currentSlideIndex;
        if (isMergedStates[currentIndex]) {
            restoreMerge();
        } else {
            mergeEdges();
        }
    });
    document.getElementById('center-btn').addEventListener('click', centerGraph);
    document.getElementById('beautiful-btn').addEventListener('click', beautifyGraph);
    document.getElementById('download-btn').addEventListener('click', downloadGraph);
    document.getElementById('evolution-btn').addEventListener('click', enterEvolutionMode);
    document.getElementById('exit-evolution-btn').addEventListener('click', exitEvolutionMode);
    document.getElementById('sla-btn').addEventListener('click', enterSLAMode);
    document.getElementById('exit-sla-btn').addEventListener('click', exitSLAMode);
    document.getElementById('label-lang-btn').addEventListener('click', toggleLabelLanguage);

    // Initialize label language button state
    const labelLangBtn = document.getElementById('label-lang-btn');
    if (labelLangBtn) {
        labelLangBtn.classList.toggle('lang-en', labelLanguage === 'en');
        labelLangBtn.textContent = (labelLanguage === 'zh') ? '中/EN' : 'EN/中';
    }

    document.getElementById('confirm-add-edge').addEventListener('click', addEdge);
    document.getElementById('cancel-add-edge').addEventListener('click', hideAddEdgeForm);
    document.getElementById('confirm-edit-edge').addEventListener('click', updateEdge);
    document.getElementById('cancel-edit-edge').addEventListener('click', hideEditEdgeForm);
    
    // Language search + dropdown
    const langInput = document.getElementById('language-search-input');
    const langResults = document.getElementById('language-search-results');
    const langBtn = document.getElementById('language-filter-btn');
    langInput.addEventListener('input', function() {
        langBtn.textContent = 'All ▼';
        searchLanguage(this.value);
    });
    langInput.addEventListener('focus', function() { searchLanguage(this.value); });
    langInput.addEventListener('blur', function() { setTimeout(() => { langResults.style.display = 'none'; }, 200); });
    langInput.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') { langResults.style.display = 'none'; }
    });
    if (langBtn) {
        langBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            const isVisible = langResults.style.display === 'block';
            if (isVisible) {
                langResults.style.display = 'none';
            } else {
                searchLanguage(langInput.value || '');
            }
        });
    }
    
    // 绑定轮播控制事件
    document.getElementById('prev-btn').addEventListener('click', showPreviousSlide);
    document.getElementById('next-btn').addEventListener('click', showNextSlide);
    
    // 绑定指示器点击事件
    const indicators = document.querySelectorAll('.indicator');
    indicators.forEach(indicator => {
        indicator.addEventListener('click', function() {
            const index = parseInt(this.getAttribute('data-index'));
            showSlide(index);
        });
    });
});

// 初始化筛选功能
function initializeFormFilter() {
    const filterBtn = document.getElementById('form-filter-btn');
    const filterMenu = document.getElementById('form-filter-menu');
    const formSearchInput = document.getElementById('form-search-input');
    
    if (!filterBtn || !filterMenu) return;
    
    filterBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        const isVisible = filterMenu.style.display === 'block';
        filterMenu.style.display = isVisible ? 'none' : 'block';
        if (!isVisible && formSearchInput) {
            filterFormMenu(formSearchInput.value);
        }
    });
    
    document.addEventListener('click', function() {
        filterMenu.style.display = 'none';
    });
    
    filterMenu.addEventListener('click', function(e) {
        e.stopPropagation();
    });
    
    if (formSearchInput) {
        formSearchInput.addEventListener('input', function() {
            filterBtn.textContent = 'All ▼';
            filterFormMenu(this.value);
            filterMenu.style.display = 'block';
        });
        formSearchInput.addEventListener('focus', function() {
            filterFormMenu(this.value);
            filterMenu.style.display = 'block';
        });
        formSearchInput.addEventListener('blur', function() {
            setTimeout(() => { filterMenu.style.display = 'none'; }, 200);
        });
    }
}

function filterFormMenu(query) {
    const filterMenu = document.getElementById('form-filter-menu');
    const items = filterMenu.querySelectorAll('.dropdown-item');
    const q = (query || '').trim().toLowerCase();
    items.forEach(item => {
        if (!q) {
            item.style.display = '';
            return;
        }
        const text = (item.textContent || '').toLowerCase();
        item.style.display = text.includes(q) ? '' : 'none';
    });
}

// 更新筛选模块
function updateFormFilter(formsData, formGroups) {
    const filterMenu = document.getElementById('form-filter-menu');
    const filterBtn = document.getElementById('form-filter-btn');
    
    if (!filterMenu || !filterBtn) {
        return;
    }
    
    // 清空现有选项
    filterMenu.innerHTML = '';
    
    // 添加"All Forms"选项
    const allOption = document.createElement('div');
    allOption.className = 'dropdown-item selected';
    allOption.textContent = 'All Forms';
    allOption.dataset.form = 'all';
    allOption.addEventListener('click', function() {
        selectFormFilter('all', 'All Forms');
    });
    filterMenu.appendChild(allOption);
    
    // 添加语组分组选项
    if (formGroups && typeof formGroups === 'object') {
        Object.keys(formGroups).forEach(groupCol => {
            const groups = formGroups[groupCol];
            groups.forEach(group => {
                const option = document.createElement('div');
                option.className = 'dropdown-item dropdown-group-item';
                option.textContent = `${groupCol}：${group.name}`;
                option.dataset.groupCol = groupCol;
                option.dataset.groupName = group.name;
                option.addEventListener('click', function() {
                    selectGroupFilter(groupCol, group.name, group.form_indices);
                });
                filterMenu.appendChild(option);
            });
        });
    }
    
    // 添加每个form选项
    formsData.forEach((formData, index) => {
        const option = document.createElement('div');
        option.className = 'dropdown-item' + (formData.isOverlay ? ' overlay-form-item' : '');
        option.textContent = formData.isOverlay 
            ? `⟐ ${formData.form} (${formData.language})` 
            : `${formData.form} (${formData.language})`;
        if (formData.isOverlay) option.title = '投影形式（不参与算法构建）';
        option.dataset.form = index;
        option.addEventListener('click', function() {
            selectFormFilter(index, `${formData.form} (${formData.language})`);
        });
        filterMenu.appendChild(option);
    });
    
    // 重置按钮和搜索框
    filterBtn.textContent = 'All ▼';
    const formSearchInput = document.getElementById('form-search-input');
    if (formSearchInput) formSearchInput.value = '';
}

// 选择筛选项（支持多选）
let selectedFormIndices = [];

function selectFormFilter(formIndex, displayText) {
    const filterBtn = document.getElementById('form-filter-btn');
    const filterMenu = document.getElementById('form-filter-menu');
    
    if (!filterBtn || !filterMenu) return;
    
    const dropdownItems = filterMenu.querySelectorAll('.dropdown-item');
    
    if (formIndex === 'all') {
        selectedFormIndices = [];
        dropdownItems.forEach(item => item.classList.remove('selected'));
        dropdownItems[0].classList.add('selected');
        filterBtn.textContent = 'All ▼';
    } else {
        const idx = selectedFormIndices.indexOf(formIndex);
        if (idx >= 0) {
            selectedFormIndices.splice(idx, 1);
        } else {
            selectedFormIndices.push(formIndex);
        }
        dropdownItems[0].classList.remove('selected');
        dropdownItems.forEach(item => {
            const fi = item.dataset.form;
            if (fi === 'all') return;
            if (selectedFormIndices.includes(parseInt(fi))) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
        if (selectedFormIndices.length === 0) {
            dropdownItems[0].classList.add('selected');
            filterBtn.textContent = 'All ▼';
        } else if (selectedFormIndices.length === 1) {
            filterBtn.textContent = displayText.split(' (')[0] + ' ▼';
        } else {
            filterBtn.textContent = selectedFormIndices.length + ' forms ▼';
        }
    }
    
    const formSearchInput = document.getElementById('form-search-input');
    if (formSearchInput) formSearchInput.value = '';
    filterFormMenu('');
    if (formIndex === 'all') {
        filterMenu.style.display = 'none';
    }
    
    // Clear language search when using form filter
    document.getElementById('language-search-input').value = '';
    const langBtn = document.getElementById('language-filter-btn');
    if (langBtn) langBtn.textContent = 'All ▼';
    currentLanguageHighlight = null;
    
    // Highlight multiple forms
    highlightFormNodes(selectedFormIndices.length > 0 ? selectedFormIndices : 'all');
}

function selectGroupFilter(groupCol, groupName, formIndices) {
    const filterBtn = document.getElementById('form-filter-btn');
    const filterMenu = document.getElementById('form-filter-menu');
    
    if (!filterBtn || !filterMenu) return;
    
    const dropdownItems = filterMenu.querySelectorAll('.dropdown-item');
    dropdownItems.forEach(item => item.classList.remove('selected'));
    dropdownItems.forEach(item => {
        if (item.dataset.groupCol === groupCol && item.dataset.groupName === groupName) {
            item.classList.add('selected');
        }
    });
    
    filterBtn.textContent = `${groupCol}：${groupName} ▼`;
    filterMenu.style.display = 'none';
    
    currentSelectedFormData = null;
    selectedFormIndices = [];
    highlightGroupNodes(formIndices);
}

function highlightFormNodes(formIndexOrIndices) {
    clearNodeHighlight();
    
    let indices = [];
    if (formIndexOrIndices === 'all') {
        return;
    } else if (Array.isArray(formIndexOrIndices)) {
        indices = formIndexOrIndices;
    } else {
        indices = [formIndexOrIndices];
    }
    
    if (!window.currentFormsData || indices.length === 0) return;
    
    const legend = document.getElementById('form-legend');
    legend.innerHTML = '';
    
    indices.forEach(formIndex => {
        const formData = window.currentFormsData[formIndex];
        if (!formData) return;
        const color = FORM_COLORS[formIndex % FORM_COLORS.length];
        
        const item = document.createElement('div');
        item.className = 'form-legend-item' + (formData.isOverlay ? ' overlay-legend' : '');
        item.style.borderColor = color;
        const dot = document.createElement('span');
        dot.className = 'form-legend-dot';
        dot.style.background = color;
        const label = document.createElement('span');
        label.textContent = (formData.isOverlay ? '⟐ ' : '') + formData.form + ' (' + formData.language + ')';
        if (formData.isOverlay) label.title = '投影形式（不参与算法构建）';
        item.appendChild(dot);
        item.appendChild(label);
        legend.appendChild(item);
        
        currentFormOutlines.push({
            formName: (formData.isOverlay ? '【古】' : '') + formData.form + ' (' + formData.language.split(' (')[0] + ')',
            color: color,
            nodeIds: formData.nodes,
            isOverlay: !!formData.isOverlay
        });
        formData.nodes.forEach(nid => highlightedNodes.push(nid));
    });
    
    legend.style.display = 'flex';
    currentSelectedFormData = indices.length === 1 ? window.currentFormsData[indices[0]] : null;
    networks.forEach(n => { if (n) n.redraw(); });
    setTimeout(updateComparisonPanel, 100);
}

function highlightGroupNodes(formIndices) {
    clearNodeHighlight();
    
    if (!window.currentFormsData || formIndices.length === 0) return;
    
    formIndices.forEach(idx => {
        if (window.currentFormsData[idx]) {
            const fd = window.currentFormsData[idx];
            const color = FORM_COLORS[idx % FORM_COLORS.length];
            currentFormOutlines.push({
                formName: (fd.isOverlay ? '【古】' : '') + fd.form + ' (' + fd.language.split(' (')[0] + ')',
                color: color,
                nodeIds: fd.nodes,
                isOverlay: !!fd.isOverlay
            });
            fd.nodes.forEach(nid => highlightedNodes.push(nid));
        }
    });
    networks.forEach(n => { if (n) n.redraw(); });
    setTimeout(updateComparisonPanel, 100);
}

function clearNodeHighlight() {
    currentFormOutlines = [];
    highlightedNodes = [];
    document.getElementById('form-legend').style.display = 'none';
    currentLanguageHighlight = null;
    outlineUserOffsets = {};
    outlineDragging = null;
    outlineHoverVertex = null;
    networks.forEach(n => { if (n) n.redraw(); });
    updateComparisonPanel();
}

// --- Connected component detection ---
function findConnectedComponents(nodeIds, edges) {
    const nodeSet = new Set(nodeIds.map(String));
    const adj = {};
    nodeSet.forEach(nid => { adj[nid] = []; });
    edges.forEach(e => {
        const f = String(e.from), t = String(e.to);
        if (nodeSet.has(f) && nodeSet.has(t)) {
            adj[f].push(t);
            adj[t].push(f);
        }
    });
    const visited = new Set();
    const components = [];
    nodeSet.forEach(nid => {
        if (visited.has(nid)) return;
        const comp = [];
        const stack = [nid];
        while (stack.length) {
            const cur = stack.pop();
            if (visited.has(cur)) continue;
            visited.add(cur);
            comp.push(cur);
            (adj[cur] || []).forEach(nb => { if (!visited.has(nb)) stack.push(nb); });
        }
        components.push(comp);
    });
    return components;
}

// --- Highlight nodes on canvas ---
function highlightNodesOnCanvas(ctx, network, idx) {
    if (highlightedNodes.length === 0) return;
    const nodeSet = new Set(highlightedNodes.map(String));
    const allNodeIds = nodeDatasets[idx] ? nodeDatasets[idx].getIds() : [];
    const targetIds = allNodeIds.filter(id => nodeSet.has(String(id)));
    if (targetIds.length === 0) return;
    const rawPos = network.getPositions(targetIds);
    const nodePositions = targetIds.map(nid => ({id: nid, ...rawPos[nid]})).filter(p => p.x !== undefined);
    
    ctx.save();
    ctx.lineWidth = 3.5;
    ctx.shadowBlur = 0;
    nodePositions.forEach(p => {
        const nodeData = nodeDatasets[idx].get(p.id);
        let outlineColor = '#FF6D00';
        for (let oi = 0; oi < currentFormOutlines.length; oi++) {
            if (currentFormOutlines[oi].nodeIds.map(String).includes(String(p.id))) {
                outlineColor = currentFormOutlines[oi].color;
                break;
            }
        }
        const drawColor = ensureNonBlue(outlineColor);
        ctx.strokeStyle = drawColor;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 26, 0, Math.PI * 2);
        ctx.stroke();
    });
    ctx.restore();
}

function ensureNonBlue(hex) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    if (b > r + 40 && b > g + 30) {
        return '#' + [
            Math.min(255, r + 80).toString(16).padStart(2, '0'),
            Math.min(255, g + 40).toString(16).padStart(2, '0'),
            Math.max(0, b - 60).toString(16).padStart(2, '0')
        ].join('');
    }
    return hex;
}

// --- Canvas drawing of outlines ---
function drawFormOutlines(ctx, network, edges) {
    const nodeRadius = 25;
    const cornerRadius = 14;
    const pad = nodeRadius + cornerRadius + 2;
    const handleRadius = 5;

    const drawItems = [];
    currentFormOutlines.forEach((outline, oi) => {
        const components = findConnectedComponents(outline.nodeIds, edges);
        components.forEach((comp, ci) => {
            const rawPos = network.getPositions(comp);
            const positions = comp.map(nid => rawPos[nid]).filter(p => p);
            if (positions.length === 0) return;

            let hull = null, topPt, isCircle = false, isPill = false;
            if (positions.length === 1) {
                isCircle = true;
                topPt = { x: positions[0].x, y: positions[0].y - pad };
            } else {
                hull = inflateNodesToHull(positions, pad);
                const key = outline.formName + '|' + outline.color;
                if (outlineUserOffsets[key] && outlineUserOffsets[key].length === hull.length) {
                    hull = hull.map((p, i) => ({
                        x: p.x + outlineUserOffsets[key][i].dx,
                        y: p.y + outlineUserOffsets[key][i].dy
                    }));
                }
                if (hull.length < 3) {
                    isPill = true;
                    topPt = { x: (hull[0].x + hull[1].x) / 2, y: Math.min(hull[0].y, hull[1].y) - pad };
                } else {
                    topPt = { x: Infinity, y: Infinity };
                    hull.forEach(p => { if (p.y < topPt.y || (p.y === topPt.y && p.x < topPt.x)) topPt = p; });
                }
            }
            outline._hull = hull;
            outline._isCircle = isCircle;
            outline._isPill = isPill;
            outline._positions = positions;
            outline._pad = pad;
            drawItems.push({ outline, oi, hull, topPt, isCircle, isPill, positions });
        });
    });

    const labelH = 18;
    drawItems.forEach(d => {
        d.labelX = d.topPt.x;
        d.labelY = d.topPt.y - 8;
    });
    drawItems.sort((a, b) => a.labelY - b.labelY);
    for (let i = 1; i < drawItems.length; i++) {
        const prevBottom = drawItems[i - 1].labelY + labelH;
        if (drawItems[i].labelY < prevBottom) {
            drawItems[i].labelY = prevBottom + 2;
        }
    }

    drawItems.forEach(d => {
        ctx.save();
        const strokeColor = ensureNonBlue(d.outline.color);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2.5;
        ctx.setLineDash(d.outline.isOverlay ? [2, 4] : [8, 5]);

        if (d.isCircle) {
            ctx.beginPath();
            ctx.arc(d.positions[0].x, d.positions[0].y, pad, 0, Math.PI * 2);
            ctx.stroke();
        } else if (d.isPill) {
            drawPillOutline(ctx, d.hull, pad);
            ctx.stroke();
        } else {
            ctx.beginPath();
            drawRoundedPolygon(ctx, d.hull, cornerRadius);
            ctx.stroke();

            ctx.setLineDash([]);
            d.hull.forEach((p, vi) => {
                const isHov = outlineHoverVertex &&
                    outlineHoverVertex.oi === d.oi && outlineHoverVertex.vi === vi;
                ctx.beginPath();
                ctx.arc(p.x, p.y, isHov ? handleRadius + 2 : handleRadius, 0, Math.PI * 2);
                ctx.fillStyle = isHov ? strokeColor : 'rgba(255,255,255,0.7)';
                ctx.fill();
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 1.5;
                ctx.stroke();
            });
        }

        ctx.setLineDash([]);
        ctx.fillStyle = strokeColor;
        ctx.font = 'bold 15px Arial';
        ctx.fillText(d.outline.formName, d.labelX - ctx.measureText(d.outline.formName).width / 2, d.labelY);
        ctx.restore();
    });
}

function drawSLALevelOutlinesOnCanvas(ctx, network, edges) {
    if (!isSLAMode || slaLevelOutlines.length === 0) return;
    const nodeRadius = 25;
    const cornerRadius = 10;
    const pad = nodeRadius + cornerRadius + 2;

    slaLevelOutlines.forEach((ol, oi) => {
        const components = findConnectedComponents(ol.nodeIds, edges);
        components.forEach((comp, ci) => {
            const rawPos = network.getPositions(comp);
            const positions = comp.map(nid => rawPos[nid]).filter(p => p);
            if (positions.length === 0) return;

            let hull = null;
            if (positions.length === 1) {
                // Single node — draw filled circle
                ctx.save();
                ctx.beginPath();
                ctx.arc(positions[0].x, positions[0].y, pad, 0, Math.PI * 2);
                ctx.fillStyle = ol.color + (ol.isActive ? '33' : '15');
                ctx.fill();
                ctx.strokeStyle = ol.color;
                ctx.lineWidth = ol.isActive ? 3 : 1.5;
                ctx.setLineDash(ol.isActive ? [] : [4, 3]);
                ctx.stroke();
                ctx.restore();
                return;
            }

            hull = inflateNodesToHull(positions, pad);
            if (hull.length < 3) {
                // Two nodes — draw pill
                ctx.save();
                ctx.beginPath();
                drawPillOutline(ctx, hull, pad);
                ctx.fillStyle = ol.color + (ol.isActive ? '33' : '15');
                ctx.fill();
                ctx.strokeStyle = ol.color;
                ctx.lineWidth = ol.isActive ? 3 : 1.5;
                ctx.setLineDash(ol.isActive ? [] : [4, 3]);
                ctx.stroke();
                ctx.restore();
                return;
            }

            ctx.save();
            ctx.beginPath();
            drawRoundedPolygon(ctx, hull, cornerRadius);
            ctx.fillStyle = ol.color + (ol.isActive ? '33' : '15');
            ctx.fill();
            ctx.strokeStyle = ol.color;
            ctx.lineWidth = ol.isActive ? 3 : 1.5;
            ctx.setLineDash(ol.isActive ? [] : [4, 3]);
            ctx.stroke();
            ctx.restore();
        });
    });
}

function inflateNodesToHull(nodes, radius) {
    const expandedPoints = [];
    const numCirclePoints = 16;
    for (const node of nodes) {
        for (let i = 0; i < numCirclePoints; i++) {
            const angle = (2 * Math.PI * i) / numCirclePoints;
            expandedPoints.push({
                x: node.x + Math.cos(angle) * radius,
                y: node.y + Math.sin(angle) * radius
            });
        }
    }
    return convexHull(expandedPoints);
}

function convexHull(points) {
    if (points.length <= 2) return points.slice();
    const sorted = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
    const cross = (O, A, B) => (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    const lower = [];
    for (const p of sorted) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
        lower.push(p);
    }
    const upper = [];
    for (let i = sorted.length - 1; i >= 0; i--) {
        const p = sorted[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    return lower.concat(upper);
}

function drawPillOutline(ctx, hull, r) {
    const A = hull[0], B = hull[hull.length - 1];
    const dx = B.x - A.x, dy = B.y - A.y;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const nx = -dy / len, ny = dx / len;
    ctx.beginPath();
    ctx.moveTo(A.x + nx * r, A.y + ny * r);
    ctx.lineTo(B.x + nx * r, B.y + ny * r);
    ctx.arc(B.x, B.y, r, Math.atan2(ny, nx), Math.atan2(-ny, -nx), false);
    ctx.lineTo(A.x - nx * r, A.y - ny * r);
    ctx.arc(A.x, A.y, r, Math.atan2(-ny, -nx), Math.atan2(ny, nx), false);
    ctx.closePath();
}

function drawRoundedPolygon(ctx, points, cornerRadius) {
    const n = points.length;
    if (n < 3) {
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < n; i++) ctx.lineTo(points[i].x, points[i].y);
        ctx.closePath();
        return;
    }
    ctx.moveTo(
        (points[n - 1].x + points[0].x) / 2,
        (points[n - 1].y + points[0].y) / 2
    );
    for (let i = 0; i < n; i++) {
        const p1 = points[i];
        const p2 = points[(i + 1) % n];
        ctx.arcTo(p1.x, p1.y, p2.x, p2.y, cornerRadius);
    }
    ctx.closePath();
}

// --- Outline vertex dragging ---
function setupOutlineDragHandling(network, idx) {
    const canvas = network.canvas.frame.canvas;
    const hitRadius = 10;

    canvas.addEventListener('mousedown', function(e) {
        if (currentFormOutlines.length === 0) return;
        const rect = canvas.getBoundingClientRect();
        const domX = e.clientX - rect.left;
        const domY = e.clientY - rect.top;
        const pos = network.DOMtoCanvas({ x: domX, y: domY });

        for (let oi = 0; oi < currentFormOutlines.length; oi++) {
            const outline = currentFormOutlines[oi];
            if (!outline._hull || outline._isCircle || outline._isPill) continue;
            const hull = outline._hull;
            for (let vi = 0; vi < hull.length; vi++) {
                const dx = pos.x - hull[vi].x, dy = pos.y - hull[vi].y;
                if (Math.sqrt(dx * dx + dy * dy) < hitRadius) {
                    const key = outline.formName + '|' + outline.color;
                    if (!outlineUserOffsets[key] || outlineUserOffsets[key].length !== hull.length) {
                        outlineUserOffsets[key] = hull.map(() => ({ dx: 0, dy: 0 }));
                    }
                    outlineDragging = {
                        oi, vi, key,
                        startMouse: { x: pos.x, y: pos.y },
                        startOffset: { ...outlineUserOffsets[key][vi] }
                    };
                    e.stopPropagation();
                    e.preventDefault();
                    return;
                }
            }
        }
    });

    canvas.addEventListener('mousemove', function(e) {
        if (outlineDragging) {
            const rect = canvas.getBoundingClientRect();
            const pos = network.DOMtoCanvas({ x: e.clientX - rect.left, y: e.clientY - rect.top });
            const d = outlineDragging;
            outlineUserOffsets[d.key][d.vi] = {
                dx: d.startOffset.dx + (pos.x - d.startMouse.x),
                dy: d.startOffset.dy + (pos.y - d.startMouse.y)
            };
            network.redraw();
            return;
        }

        const rect = canvas.getBoundingClientRect();
        const pos = network.DOMtoCanvas({ x: e.clientX - rect.left, y: e.clientY - rect.top });
        let found = null;
        for (let oi = 0; oi < currentFormOutlines.length; oi++) {
            const outline = currentFormOutlines[oi];
            if (!outline._hull || outline._isCircle || outline._isPill) continue;
            for (let vi = 0; vi < outline._hull.length; vi++) {
                const dx = pos.x - outline._hull[vi].x, dy = pos.y - outline._hull[vi].y;
                if (Math.sqrt(dx * dx + dy * dy) < hitRadius) {
                    found = { oi, vi };
                    break;
                }
            }
            if (found) break;
        }
        if (found !== outlineHoverVertex) {
            outlineHoverVertex = found;
            canvas.style.cursor = found ? 'grab' : '';
            network.redraw();
        }
    });

    canvas.addEventListener('mouseup', function() {
        if (outlineDragging) {
            canvas.style.cursor = outlineHoverVertex ? 'grab' : '';
            outlineDragging = null;
        }
    });
}

// --- Comparison Panel ---
function updateComparisonPanel() {
    const panel = document.getElementById('comparison-panel');
    if (!panel) return;
    panel.innerHTML = '';
    if (currentFormOutlines.length < 2) {
        panel.style.display = 'none';
        return;
    }
    panel.style.display = 'flex';

    currentFormOutlines.forEach((outline, idx) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'comparison-thumb';
        const title = document.createElement('div');
        title.className = 'comparison-thumb-title';
        title.textContent = (outline.isOverlay ? '⟐ ' : '') + outline.formName;
        title.style.color = ensureNonBlue(outline.color);
        wrapper.appendChild(title);
        const cvs = document.createElement('canvas');
        cvs.width = 280;
        cvs.height = 220;
        cvs.className = 'comparison-canvas';
        wrapper.appendChild(cvs);
        panel.appendChild(wrapper);
        renderComparisonThumbnail(cvs, outline, idx);
    });
}

function renderComparisonThumbnail(cvs, outline, outlineIdx) {
    const netIdx = currentSlideIndex || 0;
    if (!networks[netIdx] || !nodeDatasets[netIdx] || !edgeDatasets[netIdx]) return;

    const ctx = cvs.getContext('2d');
    const w = cvs.width, h = cvs.height;
    const allIds = nodeDatasets[netIdx].getIds();
    if (allIds.length === 0) return;
    const rawPos = networks[netIdx].getPositions(allIds);
    const positions = allIds.map(id => ({ id, ...rawPos[id] })).filter(p => p.x !== undefined);
    if (positions.length === 0) return;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    positions.forEach(p => {
        if (p.x < minX) minX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.x > maxX) maxX = p.x;
        if (p.y > maxY) maxY = p.y;
    });
    const margin = 40;
    const rangeX = (maxX - minX) || 1, rangeY = (maxY - minY) || 1;
    const scale = Math.min((w - 2 * margin) / rangeX, (h - 2 * margin) / rangeY);
    const cx = w / 2, cy = h / 2;
    const ox = (minX + maxX) / 2, oy = (minY + maxY) / 2;
    function tx(x) { return cx + (x - ox) * scale; }
    function ty(y) { return cy + (y - oy) * scale; }

    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, w, h);

    const edges = edgeDatasets[netIdx].get();
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 0.8;
    edges.forEach(e => {
        const fp = rawPos[e.from], tp = rawPos[e.to];
        if (!fp || !tp) return;
        ctx.beginPath();
        ctx.moveTo(tx(fp.x), ty(fp.y));
        ctx.lineTo(tx(tp.x), ty(tp.y));
        ctx.stroke();
    });

    const nodeSet = new Set(outline.nodeIds.map(String));
    positions.forEach(p => {
        const isHL = nodeSet.has(String(p.id));
        ctx.beginPath();
        ctx.arc(tx(p.x), ty(p.y), isHL ? 4 : 2.5, 0, Math.PI * 2);
        if (isHL) {
            ctx.fillStyle = ensureNonBlue(outline.color);
        } else {
            ctx.fillStyle = '#c0c0c0';
        }
        ctx.fill();
    });

    if (outline._hull && !outline._isCircle && !outline._isPill) {
        const scaledHull = outline._hull.map(p => ({ x: tx(p.x), y: ty(p.y) }));
        ctx.save();
        ctx.strokeStyle = ensureNonBlue(outline.color);
        ctx.lineWidth = 2;
        ctx.setLineDash(outline.isOverlay ? [2, 3] : [5, 3]);
        ctx.beginPath();
        drawRoundedPolygon(ctx, scaledHull, 6 / scale * 2);
        ctx.stroke();
        ctx.restore();
    } else if (outline._isCircle && outline._positions && outline._positions.length === 1) {
        const p = outline._positions[0];
        ctx.save();
        ctx.strokeStyle = ensureNonBlue(outline.color);
        ctx.lineWidth = 2;
        ctx.setLineDash(outline.isOverlay ? [2, 3] : [5, 3]);
        ctx.beginPath();
        ctx.arc(tx(p.x), ty(p.y), outline._pad * scale, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }
}

// --- Language Search ---
function getUniqueLanguages() {
    if (!window.currentFormsData) return [];
    const seen = new Set();
    const langs = [];
    window.currentFormsData.forEach(fd => {
        if (fd.language && !seen.has(fd.language)) {
            seen.add(fd.language);
            langs.push(fd.language);
        }
    });
    return langs;
}

function searchLanguage(query) {
    const results = document.getElementById('language-search-results');
    const allLangs = getUniqueLanguages();
    const q = (query || '').trim().toLowerCase();
    const matched = q ? allLangs.filter(l => l.toLowerCase().includes(q)) : allLangs;
    if (matched.length === 0) {
        results.style.display = 'none';
        return;
    }
    results.innerHTML = '';
    // Add "All Languages" option at top
    const allOpt = document.createElement('div');
    allOpt.className = 'language-search-item';
    allOpt.textContent = 'All Languages';
    allOpt.addEventListener('mousedown', function(e) {
        e.preventDefault();
        selectLanguage(null);
    });
    results.appendChild(allOpt);
    matched.forEach(lang => {
        const item = document.createElement('div');
        item.className = 'language-search-item';
        item.textContent = lang;
        item.addEventListener('mousedown', function(e) {
            e.preventDefault();
            selectLanguage(lang);
        });
        results.appendChild(item);
    });
    results.style.display = 'block';
}

function selectLanguage(lang) {
    const langInput = document.getElementById('language-search-input');
    const langBtn = document.getElementById('language-filter-btn');
    document.getElementById('language-search-results').style.display = 'none';
    
    if (!lang) {
        langInput.value = '';
        langBtn.textContent = 'All ▼';
        clearNodeHighlight();
        currentLanguageHighlight = null;
        return;
    }
    
    langBtn.textContent = lang.split(' (')[0] + ' ▼';
    
    // Clear form filter
    currentSelectedFormData = null;
    selectedFormIndices = [];
    document.getElementById('form-filter-btn').textContent = 'All ▼';
    document.getElementById('form-search-input').value = '';
    // Reset dropdown selected states
    const filterMenu = document.getElementById('form-filter-menu');
    if (filterMenu) {
        filterMenu.querySelectorAll('.dropdown-item').forEach(item => item.classList.remove('selected'));
        const allItem = filterMenu.querySelector('[data-form="all"]');
        if (allItem) allItem.classList.add('selected');
    }
    
    highlightLanguageForms(lang);
}

function highlightLanguageForms(lang) {
    clearNodeHighlight();
    currentLanguageHighlight = lang;
    
    if (!window.currentFormsData) return;
    
    const formsOfLang = window.currentFormsData.filter(fd => fd.language === lang);
    if (formsOfLang.length === 0) return;
    
    const legend = document.getElementById('form-legend');
    legend.innerHTML = '';
    
    formsOfLang.forEach((fd, idx) => {
        const color = FORM_COLORS[idx % FORM_COLORS.length];
        currentFormOutlines.push({
            formName: (fd.isOverlay ? '【古】' : '') + fd.form + ' (' + fd.language.split(' (')[0] + ')',
            color: color,
            nodeIds: fd.nodes,
            isOverlay: !!fd.isOverlay
        });
        fd.nodes.forEach(nid => highlightedNodes.push(nid));
        
        const item = document.createElement('div');
        item.className = 'form-legend-item' + (fd.isOverlay ? ' overlay-legend' : '');
        item.style.borderColor = color;
        const dot = document.createElement('span');
        dot.className = 'form-legend-dot';
        dot.style.background = color;
        const label = document.createElement('span');
        label.textContent = (fd.isOverlay ? '⟐ ' : '') + fd.form;
        if (fd.isOverlay) label.title = '投影形式（不参与算法构建）';
        item.appendChild(dot);
        item.appendChild(label);
        legend.appendChild(item);
    });
    legend.style.display = 'flex';
    
    networks.forEach(n => { if (n) n.redraw(); });
    setTimeout(updateComparisonPanel, 100);
}

function lightenColor(hex, factor) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    const lr = Math.round(r + (255 - r) * factor);
    const lg = Math.round(g + (255 - g) * factor);
    const lb = Math.round(b + (255 - b) * factor);
    return '#' + [lr,lg,lb].map(c => c.toString(16).padStart(2,'0')).join('');
}

// 初始化多个网络图（现在由updateGraph函数动态调用）
function initializeNetworks(count) {
    // 根据实际数量初始化网络图
    for (let i = 0; i < count; i++) {
        const container = document.getElementById(`graph-container-${i+1}`);
        
        // 确保容器存在
        if (!container) {
            console.error(`Container graph-container-${i+1} not found`);
            continue;
        }
        
        // 为每个图创建独立的数据对象
        const data = {
            nodes: nodeDatasets[i],
            edges: edgeDatasets[i]
        };
        
        // 为第一个图创建优化配置，减少性能负担
        let networkOptions = options;
        if (i === 0) {
            networkOptions = {
                ...options,
                physics: {
                    enabled: true,
                    stabilization: {
                        enabled: true,
                        iterations: 100,
                        updateInterval: 25
                    },
                    barnesHut: {
                        gravitationalConstant: -1000,
                        centralGravity: 0.1,
                        springLength: 100,
                        springConstant: 0.02
                    }
                }
            };
        }
        
        networks[i] = new vis.Network(container, data, networkOptions);
        
        setupOutlineDragHandling(networks[i], i);
        
        networks[i].on('afterDrawing', (function(idx) {
            return function(ctx) {
                const edges = edgeDatasets[idx] ? edgeDatasets[idx].get() : [];
                highlightNodesOnCanvas(ctx, networks[idx], idx);
                drawFormOutlines(ctx, networks[idx], edges);
                drawSLALevelOutlinesOnCanvas(ctx, networks[idx], edges);
            };
        })(i));
        
        // 添加选择事件监听
        networks[i].on('click', (function(index) {
            return function(params) {
                selectedNodes = params.nodes;
                selectedEdges = params.edges;
                
                // 更新当前图的索引
                currentSlideIndex = index;
            };
        })(i));
    }
    
    // 显示第一个轮播图
    if (count > 0) {
        showSlide(0);
    }
}

// 不再需要同步所有图的选择状态

// 轮播控制函数
function showSlide(index) {
    // 获取所有轮播图和指示器
    const slides = document.querySelectorAll('.carousel-slide');
    const indicators = document.querySelectorAll('.indicator');
    
    // 确保索引在有效范围内
    if (index < 0 || index >= slides.length) {
        return;
    }
    
    // 隐藏所有轮播图并移除指示器的活动状态
    slides.forEach(slide => slide.classList.remove('active'));
    indicators.forEach(indicator => indicator.classList.remove('active'));
    
    // 显示指定索引的轮播图并激活对应的指示器
    slides[index].classList.add('active');
    indicators[index].classList.add('active');
    
    // 更新当前索引
    currentSlideIndex = index;
    
    // 更新merge按钮状态
    const mergeBtn = document.getElementById('merge-edge-btn');
    const isGTSlice = (index === gtSlideIndex);
    const actionBtns = ['add-edge-btn', 'edit-edge-btn', 'delete-edge-btn', 'merge-edge-btn'];
    actionBtns.forEach(btnId => {
        const btn = document.getElementById(btnId);
        if (btn) btn.style.display = (isGTSlice || isSLAMode) ? 'none' : '';
    });
    if (mergeBtn && !isGTSlice) {
        if (isMergedStates[index]) {
            mergeBtn.textContent = 'Restore Merge';
        } else {
            mergeBtn.textContent = 'Merge Edge';
        }
    }
    
    // 更新当前显示的评价指标
    // 无论是否有评估数据，都调用updateEvaluationMetrics来确保正确显示
    updateEvaluationMetrics(index, evaluationMetricsData[index] || null);
    
    // 重新绘制当前显示的网络图
    if (networks[index]) {
        networks[index].redraw();
        // 只在初始化时或数据更新后才调用fit，避免影响用户的视图状态
        // networks[index].fit();
    }
}

function showNextSlide() {
    const slides = document.querySelectorAll('.carousel-slide');
    const nextIndex = (currentSlideIndex + 1) % slides.length;
    showSlide(nextIndex);
}

function showPreviousSlide() {
    const slides = document.querySelectorAll('.carousel-slide');
    const prevIndex = (currentSlideIndex - 1 + slides.length) % slides.length;
    showSlide(prevIndex);
}

// 处理文件上传
async function handleFileUpload() {
    const fileInput = document.getElementById('excel-file');
    const fileInfo = document.getElementById('file-info');
    
    if (isEvolutionMode) {
        exitEvolutionModeInternal();
    }
    if (isSLAMode) {
        exitSLAModeInternal();
    }
    currentExampleName = '';
    
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select an Excel file first');
        return;
    }
    
    const file = fileInput.files[0];
    fileInfo.textContent = `Selected file: ${file.name}`;
    
    try {
        // 读取Excel文件
        const data = await readExcelFile(file);
        
        // 保存原始Excel数据供merge edges功能使用
        originalExcelData = data;
        
        // 尝试发送数据到后端处理
        try {
            const response = await sendDataToBackend(data);
            // 检查返回的数据格式
            if (response.graph_data && response.forms_with_nodes) {
                // 新格式：包含graph_data和forms_with_nodes
                const graphDataArray = response.graph_data;
                const formsWithNodes = response.forms_with_nodes;
                
                // 存储forms数据供筛选功能使用
                window.currentFormsData = formsWithNodes;
                window.currentFormGroups = response.form_groups || {};
                
                updateGraph(graphDataArray);
                updateGraphTitles(file.name, graphDataArray);
                
                // 更新筛选模块
                updateFormFilter(formsWithNodes, response.form_groups || {});
            } else if (Array.isArray(response)) {
                // 兼容旧格式：直接返回数组
                updateGraph(response);
                updateGraphTitles(file.name, response);
                window.currentFormsData = [];
                window.currentFormGroups = {};
                updateFormFilter([], {});
            } else {
                // 兼容旧格式：单个语义地图
                updateGraph([response]);
                updateGraphTitles(file.name, [response]);
                window.currentFormsData = [];
                updateFormFilter([]);
            }
        } catch (error) {
            console.error('后端处理失败:', error);
            // 后端处理失败，显示错误信息
            fileInfo.textContent = `后端处理失败: ${error.message}`;
            return;
        }
    } catch (error) {
        console.error('文件处理失败:', error);
        fileInfo.textContent = `文件处理失败: ${error.message}`;
    }
}

async function handleExampleUploadGeneric(fileName) {
    // 清除上一个案例的所有状态
    clearNodeHighlight();
    currentSelectedFormData = null;
    selectedFormIndices = [];
    currentLanguageHighlight = null;
    window.currentFormsData = [];
    window.currentFormGroups = {};
    highlightedNodes = [];
    currentFormOutlines = [];
    document.getElementById('form-search-input').value = '';
    document.getElementById('form-filter-btn').textContent = 'All ▼';
    document.getElementById('language-search-input').value = '';
    const langBtn = document.getElementById('language-filter-btn');
    if (langBtn) langBtn.textContent = 'All ▼';
    document.getElementById('form-legend').style.display = 'none';

    // Exit evolution mode if active
    if (isEvolutionMode) {
        exitEvolutionModeInternal();
    }
    // Exit SLA mode if active
    if (isSLAMode) {
        exitSLAModeInternal();
    }
    evolutionData = null;
    currentExampleName = fileName;

    const fileInfo = document.getElementById('file-info');
    try {
        fileInfo.textContent = 'Loading example file...';
        const res = await fetch('data_pure/' + encodeURIComponent(fileName));
        if (!res.ok) {
            throw new Error(`Failed to load example file: ${res.status}`);
        }
        const blob = await res.blob();
        const file = new File([blob], fileName, {
            type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        });
        fileInfo.textContent = `Loaded example file: ${file.name}`;
        const data = await readExcelFile(file);
        originalExcelData = data;
        const backendRes = await sendDataToBackend(data);
        if (backendRes.graph_data && backendRes.forms_with_nodes) {
            const graphDataArray = backendRes.graph_data;
            const formsWithNodes = backendRes.forms_with_nodes;
            window.currentFormsData = formsWithNodes;
            window.currentFormGroups = backendRes.form_groups || {};
            updateGraph(graphDataArray);
            updateGraphTitles(file.name, graphDataArray);
            updateFormFilter(formsWithNodes, backendRes.form_groups || {});
        } else if (Array.isArray(backendRes)) {
            updateGraph(backendRes);
            updateGraphTitles(file.name, backendRes);
            window.currentFormsData = [];
            window.currentFormGroups = {};
            updateFormFilter([], {});
        } else {
            updateGraph([backendRes]);
            updateGraphTitles(file.name, [backendRes]);
            window.currentFormsData = [];
            window.currentFormGroups = {};
            updateFormFilter([], {});
        }
    } catch (error) {
        console.error('示例文件加载失败:', error);
        fileInfo.textContent = `示例文件加载失败: ${error.message}`;
    }
}

async function handleExampleUpload() { await handleExampleUploadGeneric('case1.xlsx'); }
async function handleExampleUpload2() { await handleExampleUploadGeneric('case2.xlsx'); }
async function handleExampleUpload3() { await handleExampleUploadGeneric('case3.xlsx'); }
async function handleExampleUpload4() { await handleExampleUploadGeneric('case4.xlsx'); }
async function handleExampleUpload5() { await handleExampleUploadGeneric('case5.xlsx'); }
async function handleExampleUpload6() { await handleExampleUploadGeneric('case6.xlsx'); }
async function handleExampleUpload7() { await handleExampleUploadGeneric('case7.xlsx'); }
async function handleExampleUpload8() { await handleExampleUploadGeneric('case8.xlsx'); }
async function handleExampleUpload9() { await handleExampleUploadGeneric('case9.xlsx'); }
async function handleExampleUpload10() { await handleExampleUploadGeneric('case10.xlsx'); }
async function handleExampleUpload11() { await handleExampleUploadGeneric('case11.xlsx'); }
async function handleExampleUpload12() { await handleExampleUploadGeneric('case12.xlsx'); }

// 处理下载指南文件
async function handleDownloadGuide() {
    try {
        // 获取guidance and examples.zip文件
        const response = await fetch('data/guidance and examples.zip');
        if (!response.ok) {
            throw new Error(`Failed to load guide file: ${response.status}`);
        }
        
        // 将响应转换为Blob
        const blob = await response.blob();
        
        // 创建下载链接
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'guidance and examples.zip';
        
        // 添加到DOM并触发下载
        document.body.appendChild(a);
        a.click();
        
        // 清理
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        console.log('Guide file downloaded successfully');
    } catch (error) {
        console.error('下载指南文件失败:', error);
        alert('下载指南文件失败，请稍后重试。');
    }
}

// 动态创建轮播图幻灯片和指示器
function createCarouselSlides(count) {
    const slidesContainer = document.querySelector('.carousel-slides');
    const indicatorsContainer = document.querySelector('.carousel-indicators');
    
    // 清空现有内容
    slidesContainer.innerHTML = '';
    indicatorsContainer.innerHTML = '';
    
    // 创建幻灯片
    for (let i = 0; i < count; i++) {
        // 创建幻灯片
        const slide = document.createElement('div');
        slide.className = `carousel-slide ${i === 0 ? 'active' : ''}`;
        
        const title = document.createElement('h3');
        title.className = 'graph-title';
        title.textContent = `Semantic Map`;
        
        const container = document.createElement('div');
        container.id = `graph-container-${i + 1}`;
        container.className = 'graph-container';
        
        slide.appendChild(title);
        slide.appendChild(container);
        slidesContainer.appendChild(slide);
        
        // 创建指示器
        const indicator = document.createElement('span');
        indicator.className = `indicator ${i === 0 ? 'active' : ''}`;
        indicator.setAttribute('data-index', i);
        indicator.addEventListener('click', () => showSlide(i));
        indicatorsContainer.appendChild(indicator);
    }
}

// 更新评价指标显示
function updateEvaluationMetrics(slideIndex, evaluationMetric) {
    console.log('Updating evaluation metrics for slide:', slideIndex, 'with data:', evaluationMetric);
    // 查找固定的评价指标面板
    const metricsPanel = document.querySelector('.evaluation-metrics-panel');
    if (!metricsPanel) {
        console.warn('Evaluation metrics panel not found');
        return;
    }
    // 如果没有评价指标数据，显示默认值
    if (!evaluationMetric) {
        console.warn('No evaluation metric data provided for slide:', slideIndex + 1);
        const metricValues = metricsPanel.querySelectorAll('.metric-value');
        metricValues.forEach(value => {
            value.textContent = 'N/A';
        });
        // 清空未连接表单显示
        updateUnconnectedForms([]);
        return;
    }
    // 更新各个指标值
    const metricMapping = {
        'acc': evaluationMetric.acc,
        'prec': evaluationMetric.prec,
        'recall': evaluationMetric.recall,
        'F1': evaluationMetric.F1,
        'productivity': evaluationMetric.productivity,
        'coverage': evaluationMetric.coverage,
        'weight_sum': evaluationMetric.weight_sum,
        'deg_mean': evaluationMetric.deg_mean,
        'deg_std': evaluationMetric.deg_std,
        'rank': evaluationMetric.rank
    };
    Object.keys(metricMapping).forEach(key => {
        const valueElement = metricsPanel.querySelector(`[data-metric="${key}"]`);
        if (valueElement) {
            const value = metricMapping[key];
            if (value !== undefined && value !== null) {
                if (key === 'rank') {
                    valueElement.textContent = typeof value === 'number' ? value : value;
                } else {
                    valueElement.textContent = typeof value === 'number' ? value.toFixed(3) : value;
                }
            } else {
                valueElement.textContent = 'N/A';
            }
        }
    });
    console.log('Evaluation metrics updated successfully for slide:', slideIndex + 1);
    // 更新 Unconnected Forms 显示
    updateUnconnectedForms(evaluationMetric.unconnected_forms || []);
}

// 更新 Unconnected Forms 显示
function updateUnconnectedForms(unconnectedForms) {
    console.log('Updating unconnected forms with data:', unconnectedForms);
    
    const container = document.getElementById('unconnected-forms-container');
    if (!container) {
        console.warn('Unconnected forms container not found');
        return;
    }
    
    // 清空容器内容
    container.innerHTML = '';
    
    // 如果没有数据，显示无数据消息
    if (!unconnectedForms || unconnectedForms.length === 0) {
        container.innerHTML = '<div class="no-data-message">No unconnected forms data available</div>';
        return;
    }
    
    // 创建表格
    const table = document.createElement('table');
    table.className = 'forms-table';
    
    // 创建表头
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    const languageHeader = document.createElement('th');
    languageHeader.textContent = 'Language';
    headerRow.appendChild(languageHeader);
    
    const formHeader = document.createElement('th');
    formHeader.textContent = 'Form';
    headerRow.appendChild(formHeader);
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // 创建表体
    const tbody = document.createElement('tbody');
    
    unconnectedForms.forEach((formData, index) => {
        const row = document.createElement('tr');
        
        // 语言列
        const languageCell = document.createElement('td');
        const languageTag = document.createElement('span');
        languageTag.className = 'language-tag';
        languageTag.textContent = formData.language || 'Unknown';
        languageCell.appendChild(languageTag);
        row.appendChild(languageCell);
        
        // Form列
        const formCell = document.createElement('td');
        formCell.textContent = formData.form || 'N/A';
        row.appendChild(formCell);
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    container.appendChild(table);
    
    console.log(`Unconnected forms updated successfully with ${unconnectedForms.length} items`);
}

// Merge Edges 功能
async function mergeEdges() {
    console.log('开始执行 Merge Edges 功能');
    
    // 检查是否有原始Excel数据
    if (!originalExcelData || originalExcelData.length === 0) {
        alert('请先上传Excel文件');
        return;
    }

    if (currentSlideIndex === gtSlideIndex) {
        alert('Cannot merge edges on the Ground Truth map');
        return;
    }
    
    // 获取当前显示的语义地图数据
    const currentIndex = currentSlideIndex;
    const currentNodes = nodeDatasets[currentIndex];
    const currentEdges = edgeDatasets[currentIndex];
    
    if (!currentNodes || !currentEdges) {
        alert('当前没有可用的语义地图数据');
        return;
    }
    
    // 获取当前显示的地图标题作为map_name
    const currentSlide = document.querySelector('.carousel-slide.active');
    const currentMapName = currentSlide ? currentSlide.querySelector('.graph-title').textContent : `Semantic Map`;
    
    // 构建当前语义地图的数据结构
    const currentGraph = {
        map_name: currentMapName,
        nodes: currentNodes.get().map(node => ({
            id: node.id,
            title: node.title || node.label,
            label: node.label
        })),
        edges: currentEdges.get().filter(edge => !edge.isGtMissing).map(edge => ({
            id: edge.id,
            from: edge.from,
            to: edge.to,
            label: edge.label || '',
            value: parseFloat(edge.title) || parseFloat(edge.label) || 1.0
        }))
    };
    
    console.log('当前语义地图数据:', currentGraph);
    console.log('原始Excel数据:', originalExcelData);
    
    try {
        // 备份merge前的数据
        const currentIndex = currentSlideIndex;
        backupEvaluationMetricsArray[currentIndex] = evaluationMetricsData[currentIndex] ? JSON.parse(JSON.stringify(evaluationMetricsData[currentIndex])) : null;
        
        // 备份当前的未连接表单数据
        const unconnectedFormsElement = document.getElementById('unconnected-forms');
        if (unconnectedFormsElement) {
            backupUnconnectedFormsArray[currentIndex] = unconnectedFormsElement.innerHTML;
        }
        
        // 显示加载提示
        const originalText = document.getElementById('merge-edge-btn').textContent;
        document.getElementById('merge-edge-btn').textContent = 'Merging...';
        document.getElementById('merge-edge-btn').disabled = true;
        
        // 调用后端merge edges接口
        const blendSettings = getCurrentBlendSettings();
        const response = await fetch('/api/merge-edges', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: originalExcelData.data,
                label: originalExcelData.label,
                graph: currentGraph,
                calc_type: blendSettings.calcType,
                blend_weight: blendSettings.blendWeight,
                llm_sim_csv: (blendSettings.calcType === 'L') ? llmSimCsvText : null
            })
        });
        
        if (!response.ok) {
            // 尝试解析后端返回的错误信息
            try {
                const errorData = await response.json();
                if (errorData.error) {
                    console.error(`Merge edges 错误 (${response.status}): ${errorData.error}`);
                    throw new Error(errorData.error);
                }
            } catch (parseError) {
                console.error(`Merge edges 错误: HTTP ${response.status}`);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        }
        
        const result = await response.json();
        console.log('Merge edges 返回结果:', result);
        
        // 处理新增的边数据 - graph_data现在是一个dict包含edges和evaluation_metric
        if (result.graph_data && result.graph_data.edges && Array.isArray(result.graph_data.edges)) {
            const newEdgesData = result.graph_data.edges;
            
            // 将新边添加到当前边数据集中，并设置高亮样式
            const newEdges = newEdgesData.map(edge => {
                const value = parseFloat(edge.value) || 1;
                
                return {
                    id: edge.id,
                    from: edge.from,
                    to: edge.to,
                    label: edge.label || '',
                    title: edge.value ? edge.value.toString() : '',
                    value: value,
                    width: Math.max(1, Math.min(10, value * 2)),
                    dashes: false,
                    color: {
                        color: '#90EE90',
                        highlight: '#90EE90',
                        hover: '#90EE90'
                    },
                    isNewMergedEdge: true
                };
            });

            // Remove dashed gray "missing GT" edges that are now covered by merged edges
            const newEdgeKeys = new Set(newEdges.map(e => `${Math.min(parseInt(e.from),parseInt(e.to))}-${Math.max(parseInt(e.from),parseInt(e.to))}`));
            const existingEdges = edgeDatasets[currentIndex].get();
            const toRemove = existingEdges
                .filter(e => e.isGtMissing && newEdgeKeys.has(`${Math.min(parseInt(e.from),parseInt(e.to))}-${Math.max(parseInt(e.from),parseInt(e.to))}`))
                .map(e => e.id);
            if (toRemove.length > 0) {
                edgeDatasets[currentIndex].remove(toRemove);
            }
            
            // 添加新边到当前数据集
            edgeDatasets[currentIndex].add(newEdges);
            
            // 更新所有边的宽度（包括新添加的边）
            updateEdgeWidths(currentIndex);
            
            // 存储新边的ID用于后续取消高亮和删除
            window.highlightedMergedEdges = newEdges.map(edge => edge.id);
            mergedEdgeIdsArray[currentIndex] = newEdges.map(edge => edge.id);
            
            // 为网络图添加点击事件监听器，点击画布时取消高亮
            if (networks[currentIndex]) {
                // 移除之前的监听器（如果存在）
                networks[currentIndex].off('click');
                
                // 添加新的点击监听器，保留原有功能并添加取消高亮功能
                networks[currentIndex].on('click', function(params) {
                    // 保留原有的选择功能
                    selectedNodes = params.nodes;
                    selectedEdges = params.edges;
                    currentSlideIndex = currentIndex;
                    
                    // 如果点击的不是边或节点（即点击画布空白处）
                    if (params.nodes.length === 0 && params.edges.length === 0) {
                        // 取消高亮新合并的边
                        if (window.highlightedMergedEdges && window.highlightedMergedEdges.length > 0) {
                            const currentEdges = edgeDatasets[currentIndex].get();
                            const updatedEdges = currentEdges.map(edge => {
                                if (window.highlightedMergedEdges.includes(edge.id)) {
                                    const restoredEdge = {
                                        ...edge,
                                        color: {
                                            color: '#FFCCE5',
                                            highlight: '#FF69B4',
                                            hover: '#FF69B4'
                                        },
                                        isNewMergedEdge: false
                                    };
                                    // Re-apply GT comparison color if available
                                    if (gtMatrix && currentIndex !== gtSlideIndex) {
                                        const fi = parseInt(edge.from);
                                        const ti = parseInt(edge.to);
                                        const gtVal = gtMatrix[fi] ? (gtMatrix[fi][ti] || 0) : 0;
                                        if (gtVal > 0) {
                                            restoredEdge.color = { color: '#81C784', highlight: '#66BB6A', hover: '#66BB6A' };
                                        } else {
                                            restoredEdge.color = { color: '#EF9A9A', highlight: '#E57373', hover: '#E57373' };
                                        }
                                    }
                                    return restoredEdge;
                                }
                                return edge;
                            });
                            
                            edgeDatasets[currentIndex].update(updatedEdges);
                            window.highlightedMergedEdges = [];
                        }
                    }
                });
                
                // 重新绘制网络图
                networks[currentIndex].redraw();
                networks[currentIndex].fit();
            }
            
            // 更新评价指标数据 - evaluation_metric现在在graph_data中
            if (result.graph_data && result.graph_data.evaluation_metric) {
                evaluationMetricsData[currentIndex] = result.graph_data.evaluation_metric;
                updateEvaluationMetrics(currentIndex, result.graph_data.evaluation_metric);
            }
            
            // 更新forms数据
            if (result.forms_with_nodes) {
                window.currentFormsData = result.forms_with_nodes;
                updateFormFilter(result.forms_with_nodes);
            }
            
            // 更新Unconnected Forms - 使用evaluation_metric中的unconnected_forms字段
            if (result.graph_data && result.graph_data.evaluation_metric && result.graph_data.evaluation_metric.unconnected_forms) {
                updateUnconnectedForms(result.graph_data.evaluation_metric.unconnected_forms);
            } else if (result.forms_with_nodes) {
                // 如果没有unconnected_forms字段，则使用forms_with_nodes作为备选
                updateUnconnectedForms(result.forms_with_nodes);
            }
            
            // 更新按钮状态
            isMergedStates[currentIndex] = true;
            document.getElementById('merge-edge-btn').textContent = 'Restore Merge';
            document.getElementById('merge-edge-btn').disabled = false;
            
            console.log(`Merge edges 完成！已添加 ${newEdgesData.length} 条新边，点击画布空白处可取消高亮。`);
            return; // 成功完成，不执行finally中的按钮恢复
        }
        
    } catch (error) {
        console.error('Merge edges 时出错:', error);
        console.error('Merge edges 失败: ' + error.message);
    } finally {
        // 恢复按钮状态
        document.getElementById('merge-edge-btn').textContent = originalText;
        document.getElementById('merge-edge-btn').disabled = false;
    }
}

// 还原merge操作
function restoreMerge() {
    try {
        const currentIndex = currentSlideIndex;
        
        // 显示加载提示
        document.getElementById('merge-edge-btn').textContent = 'Restoring...';
        document.getElementById('merge-edge-btn').disabled = true;
        
        // 删除所有merge的边
        if (mergedEdgeIdsArray[currentIndex] && mergedEdgeIdsArray[currentIndex].length > 0) {
            const removedEdges = edgeDatasets[currentIndex].get(mergedEdgeIdsArray[currentIndex]);
            edgeDatasets[currentIndex].remove(mergedEdgeIdsArray[currentIndex]);

            // Re-add dashed gray "missing GT" edges for removed merged edges that are in GT but not in pred
            if (gtMatrix && currentIndex !== gtSlideIndex) {
                const currentEdges = edgeDatasets[currentIndex].get();
                const existingKeys = new Set(currentEdges.filter(e => !e.isGtMissing).map(e => `${Math.min(parseInt(e.from),parseInt(e.to))}-${Math.max(parseInt(e.from),parseInt(e.to))}`));
                const missingEdges = [];
                removedEdges.forEach(edge => {
                    const a = parseInt(edge.from);
                    const b = parseInt(edge.to);
                    const key = `${Math.min(a,b)}-${Math.max(a,b)}`;
                    const gtVal = gtMatrix[a] ? (gtMatrix[a][b] || 0) : 0;
                    if (gtVal > 0 && !existingKeys.has(key)) {
                        missingEdges.push({
                            id: `gt-missing-${a}-${b}-${currentIndex}`,
                            from: String(a), to: String(b),
                            label: '', value: 0.5, width: 1,
                            color: { color: '#BDBDBD', highlight: '#9E9E9E', hover: '#9E9E9E', opacity: 0.5 },
                            dashes: [5, 5],
                            isGtMissing: true
                        });
                    }
                });
                if (missingEdges.length > 0) {
                    edgeDatasets[currentIndex].add(missingEdges);
                }
            }

            // 重新绘制网络图
            if (networks[currentIndex]) {
                networks[currentIndex].redraw();
                networks[currentIndex].fit();
            }
        }
        
        // 还原评价指标
        if (backupEvaluationMetricsArray[currentIndex]) {
            evaluationMetricsData[currentIndex] = JSON.parse(JSON.stringify(backupEvaluationMetricsArray[currentIndex]));
            updateEvaluationMetrics(currentIndex, backupEvaluationMetricsArray[currentIndex]);
        }
        
        // 还原未连接表单
        if (backupUnconnectedFormsArray[currentIndex]) {
            const unconnectedFormsElement = document.getElementById('unconnected-forms');
            if (unconnectedFormsElement) {
                unconnectedFormsElement.innerHTML = backupUnconnectedFormsArray[currentIndex];
            }
        }
        
        // 恢复原有的click事件监听器
        if (networks[currentIndex]) {
            networks[currentIndex].off('click');
            networks[currentIndex].on('click', (function(index) {
                return function(params) {
                    selectedNodes = params.nodes;
                    selectedEdges = params.edges;
                    
                    // 更新当前图的索引
                    currentSlideIndex = index;
                };
            })(currentIndex));
        }
        
        // 清理状态
        isMergedStates[currentIndex] = false;
        mergedEdgeIdsArray[currentIndex] = [];
        window.highlightedMergedEdges = [];
        backupEvaluationMetricsArray[currentIndex] = null;
        backupUnconnectedFormsArray[currentIndex] = null;
        
        // 恢复按钮状态
        document.getElementById('merge-edge-btn').textContent = 'Merge Edge';
        document.getElementById('merge-edge-btn').disabled = false;
        
        console.log('Merge操作已还原！');
        
    } catch (error) {
        console.error('还原merge时出错:', error);
        console.error('还原merge失败: ' + error.message);
        
        // 恢复按钮状态
        document.getElementById('merge-edge-btn').textContent = 'Restore Merge';
        document.getElementById('merge-edge-btn').disabled = false;
    }
}

// 更新所有图的标题
function updateGraphTitles(fileName, graphDataArray) {
    // 从文件名中提取基本名称（不包含扩展名）
    const baseName = fileName.replace(/\.[^\.]+$/, '');
    
    // 更新每个图的标题
    const titles = document.querySelectorAll('.graph-title');
    titles.forEach((title, index) => {
        if (graphDataArray && graphDataArray[index] && graphDataArray[index].map_name) {
            title.textContent = `${graphDataArray[index].map_name}: ${baseName}`;
        } else {
            title.textContent = `语义地图 ${index + 1}: ${baseName}`;
        }
    });
}

// 读取Excel文件
function readExcelFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            try {
                const data = e.target.result;
                const workbook = XLSX.read(data, { type: 'array' });
                
                // 获取第一个工作表
                const firstSheetName = workbook.SheetNames[0];
                const firstWorksheet = workbook.Sheets[firstSheetName];
                
                // 将第一个工作表转换为JSON
                const firstSheetData = XLSX.utils.sheet_to_json(firstWorksheet, { defval: null });
                
                // 获取第二个工作表（如果存在）
                let secondSheetData = [];
                if (workbook.SheetNames.length > 1) {
                    const secondSheetName = workbook.SheetNames[1];
                    const secondWorksheet = workbook.Sheets[secondSheetName];
                    secondSheetData = XLSX.utils.sheet_to_json(secondWorksheet, { defval: null });
                }
                
                // 获取第三个工作表（如果存在）— 功能中英文对照表
                let thirdSheetData = [];
                if (workbook.SheetNames.length > 2) {
                    const thirdSheetName = workbook.SheetNames[2];
                    const thirdWorksheet = workbook.Sheets[thirdSheetName];
                    thirdSheetData = XLSX.utils.sheet_to_json(thirdWorksheet, { defval: null, header: 1 });
                }
                
                // 获取第四个工作表（如果存在）— 时序数据
                let fourthSheetData = [];
                if (workbook.SheetNames.length > 3) {
                    const fourthSheetName = workbook.SheetNames[3];
                    const fourthWorksheet = workbook.Sheets[fourthSheetName];
                    fourthSheetData = XLSX.utils.sheet_to_json(fourthWorksheet, { defval: null, header: 1 });
                }
                
                // 获取第五个工作表（如果存在）— SLA学习者水平数据
                let fifthSheetData = [];
                if (workbook.SheetNames.length > 4) {
                    const fifthSheetName = workbook.SheetNames[4];
                    const fifthWorksheet = workbook.Sheets[fifthSheetName];
                    fifthSheetData = XLSX.utils.sheet_to_json(fifthWorksheet, { defval: null, header: 1 });
                }
                
                // 返回包含五个sheet数据的对象
                resolve({
                    data: firstSheetData,
                    label: secondSheetData,
                    sheet3: thirdSheetData,
                    sheet4: fourthSheetData,
                    sheet5: fifthSheetData
                });
            } catch (error) {
                reject(error);
            }
        };
        
        reader.onerror = function(error) {
            reject(error);
        };
        
        reader.readAsArrayBuffer(file);
    });
}

// Get current blend settings
function getCurrentBlendSettings() {
    const activeBtn = document.querySelector('.blend-type-btn.active');
    const calcType = activeBtn ? activeBtn.dataset.calcType : 'J';
    const weight = parseFloat(document.getElementById('blend-weight-slider').value);
    return { calcType: calcType, blendWeight: weight };
}

// Toggle feature label language between Chinese and English
function toggleLabelLanguage() {
    labelLanguage = (labelLanguage === 'zh') ? 'en' : 'zh';
    localStorage.setItem('labelLanguage', labelLanguage);
    const btn = document.getElementById('label-lang-btn');
    if (btn) {
        btn.classList.toggle('lang-en', labelLanguage === 'en');
        btn.textContent = (labelLanguage === 'zh') ? '中/EN' : 'EN/中';
    }
    // Re-render graphs with the new language if data is already loaded
    if (originalExcelData && originalExcelData.data) {
        refetchGraph();
    }
}

// 发送数据到后端
async function sendDataToBackend(excelData) {
    try {
        const blendSettings = getCurrentBlendSettings();
        const response = await fetch('/api/process-excel', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                data: excelData.data,
                label: excelData.label,
                sheet3: excelData.sheet3 || [],
                calc_type: blendSettings.calcType,
                blend_weight: blendSettings.blendWeight,
                llm_sim_csv: (blendSettings.calcType === 'L') ? llmSimCsvText : null
            })
        });
        
        if (!response.ok) {
            // Show only ONE alert, without fixed prefix or status code
            let alertMsg = 'Request failed. Please try again later.';
            try {
                const errorData = await response.json();
                if (errorData && typeof errorData.error === 'string' && errorData.error.trim()) {
                    alertMsg = errorData.error.trim();
                }
            } catch (parseError) {
                // Ignore parse error, keep generic message
            }
            alert(alertMsg);
            throw new Error(alertMsg);
        }
        
        const jsonResponse = await response.json();
        gtMatrix = jsonResponse.gt_matrix || null;
        perExampleLabels = jsonResponse.per_example_labels || {};
        return jsonResponse;
    } catch (error) {
        console.error('发送数据到后端时出错:', error);
        
        // Network/connection errors: show ONE English alert
        if (error.message && error.message.includes('fetch')) {
            alert('Unable to connect to backend server. Request failed');
        }
        
        // Re-throw so caller can handle without duplicating alerts
        throw error;
    }
}



// 更新图形
function updateGraph(graphDataArray) {
    console.log('开始更新图形，接收到的数据:', graphDataArray);
    console.log('语义地图数量:', graphDataArray.length);
    
    // 清除现有的网络图实例
    for (let i = 0; i < networks.length; i++) {
        if (networks[i]) {
            networks[i].destroy();
            networks[i] = null;
        }
    }
    
    // 清空数组
    networks = [];
    nodeDatasets = [];
    edgeDatasets = [];
    evaluationMetricsData = [];
    
    // 初始化merge相关的状态数组
    isMergedStates = [];
    backupEvaluationMetricsArray = [];
    backupUnconnectedFormsArray = [];
    mergedEdgeIdsArray = [];
    
    // 为每个地图初始化merge状态
    for (let i = 0; i < graphDataArray.length; i++) {
        isMergedStates[i] = false;
        backupEvaluationMetricsArray[i] = null;
        backupUnconnectedFormsArray[i] = null;
        mergedEdgeIdsArray[i] = [];
    }
    
    // 动态创建轮播图幻灯片和指示器
    createCarouselSlides(graphDataArray.length);
    
    // 根据返回的数据创建对应数量的网络图
    for (let i = 0; i < graphDataArray.length; i++) {
        const graphData = graphDataArray[i];
        
        // 确保数据格式正确
        const nodes = (graphData.nodes || []).map(node => {
            // Try per-example labels (from Sheet3) first, then global supplementaryFeatureLabels
            let labelInfo = perExampleLabels[node.label] || null;
            if (!labelInfo) {
                labelInfo = supplementaryFeatureLabels[node.label];
            }
            if (labelInfo) {
                const displayLabel = (labelLanguage === 'en')
                    ? (labelInfo.english || labelInfo.chinese || node.label)
                    : (labelInfo.chinese || labelInfo.english || node.label);
                return {
                    ...node,
                    originalLabel: node.label,
                    label: displayLabel,
                    title: labelInfo.english ? (labelInfo.english + (labelInfo.abbrev ? ' (' + labelInfo.abbrev + ')' : '')) : displayLabel
                };
            }
            return node;
        });
        const isGTSlice = graphData.is_gt === true;
        const edges = (graphData.edges || []).map(edge => {
            // 确保每条边都有label属性
            if (!edge.hasOwnProperty('label') && edge.value !== undefined) {
                edge.label = edge.value.toString();
            }
            // 确保每条边都有width属性
            if (!edge.hasOwnProperty('width') && edge.value !== undefined) {
                const value = parseFloat(edge.value) || 1;
                edge.width = Math.max(1, Math.min(10, value * 2));
            }
            // GT slice: keep default pink; predicted slices: color by GT comparison
            if (!isGTSlice && gtMatrix) {
                const fi = parseInt(edge.from);
                const ti = parseInt(edge.to);
                const gtVal = gtMatrix[fi] ? (gtMatrix[fi][ti] || 0) : 0;
                if (gtVal > 0) {
                    edge.color = { color: '#81C784', highlight: '#66BB6A', hover: '#66BB6A' };
                } else {
                    edge.color = { color: '#EF9A9A', highlight: '#E57373', hover: '#E57373' };
                }
            }
            return edge;
        });

        // Add missing GT edges (in GT but not predicted) as dashed gray edges
        if (!isGTSlice && gtMatrix) {
            const existingEdgeKeys = new Set(edges.map(e => `${Math.min(parseInt(e.from),parseInt(e.to))}-${Math.max(parseInt(e.from),parseInt(e.to))}`));
            for (let a = 0; a < gtMatrix.length; a++) {
                for (let b = a + 1; b < gtMatrix[a].length; b++) {
                    if (gtMatrix[a][b] > 0) {
                        const key = `${a}-${b}`;
                        if (!existingEdgeKeys.has(key)) {
                            edges.push({
                                id: `gt-missing-${a}-${b}-${i}`,
                                from: String(a),
                                to: String(b),
                                label: gtMatrix[a][b].toFixed(2),
                                value: 0.5,
                                width: 1,
                                color: { color: '#BDBDBD', highlight: '#9E9E9E', hover: '#9E9E9E', opacity: 0.5 },
                                font: { color: '#BDBDBD', size: 10, strokeWidth: 0, opacity: 0.6 },
                                dashes: [5, 5],
                                isGtMissing: true
                            });
                        }
                    }
                }
            }
        }
        
        console.log(`处理第${i+1}个语义地图:`, graphData.map_name || `地图${i+1}`);
        console.log(`节点数量: ${nodes.length}, 边数量: ${edges.length}`);
        
        // 存储评价指标数据
        evaluationMetricsData[i] = graphData.evaluation_metric || null;
        if (graphData.is_gt === true) {
            gtSlideIndex = i;
        }
        
        nodeDatasets[i] = new vis.DataSet(nodes);
        edgeDatasets[i] = new vis.DataSet(edges);
        
        // 获取容器
        const container = document.getElementById(`graph-container-${i+1}`);
        
        // 创建数据对象
        const data = {
            nodes: nodeDatasets[i],
            edges: edgeDatasets[i]
        };
        
        // 为第一个图创建优化配置，减少性能负担
        let networkOptions = options;
        if (i === 0) {
            networkOptions = {
                ...options,
                physics: {
                    enabled: true,
                    stabilization: {
                        enabled: true,
                        iterations: 100,
                        updateInterval: 25
                    },
                    barnesHut: {
                        gravitationalConstant: -1000,
                        centralGravity: 0.1,
                        springLength: 100,
                        springConstant: 0.02
                    }
                }
            };
        }
        
        // 创建网络图
        networks[i] = new vis.Network(container, data, networkOptions);
        
        setupOutlineDragHandling(networks[i], i);
        
        // Draw form outlines on canvas
        networks[i].on('afterDrawing', (function(idx) {
            return function(ctx) {
                const edges = edgeDatasets[idx] ? edgeDatasets[idx].get() : [];
                highlightNodesOnCanvas(ctx, networks[idx], idx);
                drawFormOutlines(ctx, networks[idx], edges);
                drawSLALevelOutlinesOnCanvas(ctx, networks[idx], edges);
            };
        })(i));
        
        // 添加选择事件监听
        networks[i].on('click', (function(index) {
            return function(params) {
                selectedNodes = params.nodes;
                selectedEdges = params.edges;
                currentSlideIndex = index;
            };
        })(i));
        
        // Hover tooltip for examples when nodes are highlighted
        networks[i].on('hoverNode', (function(idx) {
            return function(params) {
                if (highlightedNodes.length === 0) return;
                const nodeId = params.node;
                if (!highlightedNodes.includes(String(nodeId))) return;
                const nodeData = nodeDatasets[idx].get(nodeId);
                if (!nodeData) return;

                // ===== Evolution mode: aggregate corpus examples for this feature =====
                if (isEvolutionMode && evolutionData && evolutionData.corpus) {
                    const html = buildEvolutionNodeTooltip(nodeData);
                    if (html) {
                        const tooltip = document.getElementById('example-tooltip');
                        tooltip.innerHTML = html;
                        tooltip.style.display = 'block';
                        const maxW = 460;
                        const left = Math.min(params.event.pageX + 15, window.innerWidth - (maxW + 20));
                        const top = Math.min(params.event.pageY + 15, window.innerHeight - 360);
                        tooltip.style.left = left + 'px';
                        tooltip.style.top = top + 'px';
                        return;
                    }
                    return;
                }

                const featureLabel = nodeData.label;
                const featureOriginal = nodeData.originalLabel || featureLabel;
                
                // Find which form(s) contain this node
                let formData = currentSelectedFormData;
                if (!formData || !formData.nodes.includes(String(nodeId))) {
                    // Try finding from all forms (language highlight mode)
                    formData = null;
                    if (window.currentFormsData) {
                        for (const fd of window.currentFormsData) {
                            if (fd.nodes.includes(String(nodeId))) {
                                formData = fd;
                                break;
                            }
                        }
                    }
                }
                if (!formData) return;
                
                let exampleText = null;
                const langBase = formData.language ? formData.language.split(' (')[0].split('（')[0] : '';
                for (const fKey of [featureLabel, featureOriginal]) {
                    if (langBase) {
                        const exKey1 = langBase + '|' + formData.form + '|' + fKey;
                        if (supplementaryFeatureExamples[exKey1]) {
                            exampleText = supplementaryFeatureExamples[exKey1];
                            break;
                        }
                    }
                    const exKey2 = formData.form + '|' + fKey;
                    if (supplementaryFeatureExamples[exKey2]) {
                        exampleText = supplementaryFeatureExamples[exKey2];
                        break;
                    }
                }
                if (exampleText) {
                    const tooltip = document.getElementById('example-tooltip');
                    const escaped = formData.form.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                    const re = new RegExp(escaped, 'g');
                    tooltip.innerHTML = exampleText.replace(/\n/g, '<br>').replace(re, '<b style="color:#e53935;">$&</b>');
                    tooltip.style.display = 'block';
                    const left = Math.min(params.event.pageX + 15, window.innerWidth - 420);
                    const top = Math.min(params.event.pageY + 15, window.innerHeight - 320);
                    tooltip.style.left = left + 'px';
                    tooltip.style.top = top + 'px';
                }
            };
        })(i));
        
        networks[i].on('blurNode', function() {
            document.getElementById('example-tooltip').style.display = 'none';
        });
        
        // 更新评价指标
        updateEvaluationMetrics(i, graphData.evaluation_metric);
        
        // 重新绘制网络图
        networks[i].redraw();
        // 只在数据更新后调用一次fit，避免重复调用影响性能
        if (i === 0) {
            // 对第一个图使用更快的fit，避免卡顿
            setTimeout(() => {
                networks[i].fit();
            }, 100);
        } else {
            // 对其他图延迟调用fit
            setTimeout(() => {
                networks[i].fit();
            }, 200 + i * 100);
        }
    }
    
    // 更新总幻灯片数量
    totalSlides = graphDataArray.length;
    
    // 显示第一个轮播图
    showSlide(0);
}

// 更新轮播图指示器
function updateCarouselIndicators(count) {
    const indicatorsContainer = document.querySelector('.carousel-indicators');
    indicatorsContainer.innerHTML = '';
    
    for (let i = 0; i < count; i++) {
        const indicator = document.createElement('span');
        indicator.className = i === 0 ? 'indicator active' : 'indicator';
        indicator.setAttribute('data-index', i);
        indicator.addEventListener('click', function() {
            const index = parseInt(this.getAttribute('data-index'));
            showSlide(index);
        });
        indicatorsContainer.appendChild(indicator);
    }
}



// 显示添加边表单
function showAddEdgeForm() {
    // 更新节点选择器
    updateNodeSelectors();
    
    // 如果当前地图的节点数量不足，显示提示
    const currentNodes = nodeDatasets[currentSlideIndex].get();
    if (currentNodes.length < 2) {
        alert('需要至少两个节点才能添加边');
        return;
    }
    
    document.getElementById('edge-form').style.display = 'block';
}

// 更新节点选择器
function updateNodeSelectors() {
    const fromSelect = document.getElementById('from-node');
    const toSelect = document.getElementById('to-node');
    
    // 清空现有选项
    fromSelect.innerHTML = '';
    toSelect.innerHTML = '';
    
    // 添加当前地图的节点选项
    nodeDatasets[currentSlideIndex].forEach(node => {
        const fromOption = document.createElement('option');
        fromOption.value = node.id;
        fromOption.textContent = node.label;
        fromSelect.appendChild(fromOption);
        
        const toOption = document.createElement('option');
        toOption.value = node.id;
        toOption.textContent = node.label;
        toSelect.appendChild(toOption);
    });
}

// 隐藏添加边表单
function hideAddEdgeForm() {
    document.getElementById('edge-form').style.display = 'none';
    document.getElementById('edge-label').value = '';
}

// 添加新边
async function addEdge() {
    const fromNode = document.getElementById('from-node').value;
    const toNode = document.getElementById('to-node').value;
    const label = document.getElementById('edge-label').value.trim();
    
    if (fromNode && toNode) {
        // 检查是否已存在相同的边（只在当前地图中检查）
        const existingEdges = edgeDatasets[currentSlideIndex].get({
            filter: function(edge) {
                return edge.from === fromNode && edge.to === toNode;
            }
        });
        
        if (existingEdges.length > 0) {
            alert('这两个节点之间已经存在边');
            return;
        }
        
        // 解析边的值（权重）
        const value = parseFloat(label) || 1;
        
        const newEdge = {
            id: `e${Date.now()}`,
            from: fromNode,
            to: toNode,
            label: label,
            value: value,
            arrows: { to: { enabled: false } },
            color: {
                color: '#FFCCE5',       // 浅粉色
                highlight: '#FF69B4',   // 深粉色
                hover: '#FF69B4'        // 鼠标悬停时显示深粉色
            }
        };
        
        // 只在当前地图上添加边
        edgeDatasets[currentSlideIndex].add(newEdge);
        hideAddEdgeForm();
        
        // Only redraw current map
        if (networks[currentSlideIndex]) {
            networks[currentSlideIndex].redraw();
        }
        
        // 更新当前地图边的宽度
        updateEdgeWidths(currentSlideIndex);
        
        // 调用后端接口进行校验
        await validateGraphWithBackend();
    }
}



// 删除选中的边
async function deleteSelectedEdge() {
    if (selectedEdges.length > 0) {
        // 确认删除
        if (confirm(`确定要删除选中的 ${selectedEdges.length} 条边吗？`)) {
            // 只从当前地图中删除边
            edgeDatasets[currentSlideIndex].remove(selectedEdges);
            selectedEdges = [];
            
            // 只重新绘制当前地图
            if (networks[currentSlideIndex]) {
                networks[currentSlideIndex].redraw();
            }
            
            // 调用后端接口进行校验
            await validateGraphWithBackend();
        }
    } else {
        alert('请先选择要删除的边');
    }
}

// 显示修改边表单
function showEditEdgeForm() {
    if (selectedEdges.length !== 1) {
        alert('请先选择一条要修改的边');
        return;
    }
    
    // 获取选中的边（从当前地图的数据集中获取）
    const edgeId = selectedEdges[0];
    const edge = edgeDatasets[currentSlideIndex].get(edgeId);
    
    if (!edge) {
        alert('无法获取选中的边');
        return;
    }
    
    // 设置当前正在编辑的边ID
    currentEditEdgeId = edgeId;
    
    // Show edit edge form
    document.getElementById('edit-edge-form').style.display = 'block';
    document.getElementById('edge-form').style.display = 'none';
    
    // 设置表单初始值
    document.getElementById('edit-edge-label').value = edge.label || '';
    document.getElementById('edit-edge-label').focus();
}

// 隐藏修改边表单
function hideEditEdgeForm() {
    document.getElementById('edit-edge-form').style.display = 'none';
    document.getElementById('edit-edge-label').value = '';
    currentEditEdgeId = null;
}

// 更新边
async function updateEdge() {
    if (currentEditEdgeId === null) {
        alert('没有选择要修改的边');
        return;
    }
    
    const newLabel = document.getElementById('edit-edge-label').value.trim();
    
    // Parse edge value (weight)
    const value = parseFloat(newLabel) || 1;
    
    // 获取当前边的信息（从当前地图的数据集中获取）
    const edge = edgeDatasets[currentSlideIndex].get(currentEditEdgeId);
    
    // 只更新当前地图的边
    edgeDatasets[currentSlideIndex].update({
        id: currentEditEdgeId,
        from: edge.from,
        to: edge.to,
        label: newLabel,
        title: newLabel,
        value: value,
        arrows: { to: { enabled: false } },
        color: {
            color: '#FFCCE5',       // 浅粉色
            highlight: '#FF69B4',   // 深粉色
            hover: '#FF69B4'        // 鼠标悬停时显示深粉色
        }
    });
    
    // 隐藏表单
    hideEditEdgeForm();
    
    // Only redraw current map
    if (networks[currentSlideIndex]) {
        networks[currentSlideIndex].redraw();
    }
    
    // 只更新当前地图边的宽度
    updateEdgeWidths(currentSlideIndex);
    
    // 调用后端接口进行校验
    await validateGraphWithBackend();
}

// 将当前显示的网络图居中
function centerGraph() {
    // 获取当前显示的网络图实例
    const network = networks[currentSlideIndex];
    
    if (network) {
        // 使用fit()方法将视图调整到适合所有节点的大小
        network.fit({
            animation: {
                duration: 1000,
                easingFunction: 'easeInOutQuad'
            }
        });
    }
}

// 美化图形布局 / 恢复原始布局
function beautifyGraph() {
    const beautifyBtn = document.getElementById('beautiful-btn');
    
    // 检查是否有网络图实例
    if (networks.length > 0) {
        if (!isBeautified) {
            // 执行美化操作
            console.log('开始美化所有图形布局');
            
            // 创建美化的配置选项
            const beautifyOptions = {
                edges: {
                    smooth: {
                        enabled: false  // 禁用平滑曲线，使边变成直线
                    }
                },
                physics: {
                    enabled: false  // 禁用物理引擎，固定节点位置
                },
                layout: {
                    improvedLayout: true,
                    clusterThreshold: 150
                },
                interaction: {
                    dragNodes: true,  // 允许拖拽节点
                    dragView: true,   // 允许拖拽视图
                    zoomView: true    // 允许缩放视图
                }
            };
            
            // 对所有网络图实例应用美化配置
            networks.forEach((network, index) => {
                if (network) {
                    // 应用美化配置
                    network.setOptions(beautifyOptions);
                    
                    console.log(`图形 ${index + 1} 美化布局完成 - 节点位置已固定`);
                    
                    // 调整视图
                    network.fit({
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            });
            
            console.log('所有图形美化布局完成 - 拖拽时其他节点将保持固定位置');
            
            // 更新状态和按钮文本
            isBeautified = true;
            beautifyBtn.textContent = 'Restore';
            
        } else {
            // 执行恢复操作
            console.log('恢复所有原始图形布局');
            
            // 创建恢复的配置选项（恢复到原始的物理引擎设置）
            const restoreOptions = {
                edges: {
                    smooth: {
                        enabled: true,  // 启用平滑曲线，允许边弯曲
                        type: 'dynamic',
                        roundness: 0.5
                    }
                },
                physics: {
                    enabled: true,
                    solver: 'barnesHut',  // 恢复到原始的barnesHut算法
                    barnesHut: {
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 150,
                        springConstant: 0.04
                    },
                    stabilization: {
                        enabled: true,
                        iterations: 1000,
                        updateInterval: 25
                    }
                },
                layout: {
                    improvedLayout: false
                },
                interaction: {
                    dragNodes: true,  // 允许拖拽节点
                    dragView: true,   // 允许拖拽视图
                    zoomView: true    // 允许缩放视图
                }
            };
            
            // 对所有网络图实例应用恢复配置
            let completedCount = 0;
            networks.forEach((network, index) => {
                if (network) {
                    // 应用恢复配置
                    network.setOptions(restoreOptions);
                    
                    // 重新稳定化网络
                    network.stabilize();
                    
                    // 稳定化完成后调整视图
                    network.once('stabilizationIterationsDone', function() {
                        completedCount++;
                        console.log(`图形 ${index + 1} 恢复布局完成`);
                        
                        // 调整视图
                        network.fit({
                            animation: {
                                duration: 1500,
                                easingFunction: 'easeInOutQuad'
                            }
                        });
                        
                        // 当所有网络都完成恢复时
                        if (completedCount === networks.length) {
                            console.log('所有图形恢复布局完成');
                        }
                    });
                }
            });
            
            // 更新状态和按钮文本
            isBeautified = false;
            beautifyBtn.textContent = 'Beautify';
        }
    }
}

// 下载图形为图片
function downloadGraph() {
    // 获取当前显示的图形容器
    const currentSlide = document.querySelector('.carousel-slide.active');
    const graphTitle = currentSlide.querySelector('.graph-title').textContent;
    const container = currentSlide.querySelector('.graph-container');
    
    // 添加下载状态类来隐藏控制按钮
    container.classList.add('downloading');
    
    // 等待CSS过渡完成后再截图
    setTimeout(() => {
        // 使用html2canvas将容器转换为canvas
        html2canvas(container, {
            ignoreElements: function(element) {
                // 忽略包含vis-相关类名的元素
                return element.className && (
                    element.className.includes('vis-navigation') ||
                    element.className.includes('vis-manipulation') ||
                    element.className.includes('vis-button')
                );
            },
            useCORS: true,
            allowTaint: true,
            scale: 2 // 提高图片质量
        }).then(canvas => {
            // 移除下载状态类，恢复按钮显示
            container.classList.remove('downloading');
            
            // 创建下载链接
            const link = document.createElement('a');
            link.download = graphTitle + '_' + new Date().toISOString().slice(0, 10) + '.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }).catch(error => {
            // 如果出错，也要移除下载状态类
            container.classList.remove('downloading');
            console.error('下载图片时出错:', error);
        });
    }, 350); // 等待CSS过渡完成
}

// 调用后端接口校验语义地图质量
async function validateGraphWithBackend() {
    try {
        if (!originalExcelData || !originalExcelData.data) {
            console.warn('没有原始Excel数据，无法进行校验');
            return;
        }
        
        // 构建当前地图的图数据
        const currentNodes = nodeDatasets[currentSlideIndex].get();
        const currentEdges = edgeDatasets[currentSlideIndex].get();
        
        // 获取当前地图的名称
        const currentSlide = document.querySelector('.carousel-slide.active');
        const mapName = currentSlide ? currentSlide.querySelector('.graph-title').textContent : `Semantic Map`;
        
        const graphData = {
            nodes: currentNodes,
            edges: currentEdges,
            map_name: mapName
        };
        
        // 调用后端接口
        const response = await fetch('/api/edge-modify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                data: originalExcelData.data,
                label: originalExcelData.label,
                graph: graphData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const evaluationResult = await response.json();
        
        // 更新评价指标数据
        evaluationMetricsData[currentSlideIndex] = evaluationResult;
        
        // 更新评价指标显示
        updateEvaluationMetrics(currentSlideIndex, evaluationResult);
        
        console.log('语义地图校验完成，评价指标已更新');
        
    } catch (error) {
        console.error('调用后端校验接口时出错:', error);
        // 可以选择显示错误提示，但不阻断用户操作
        // alert('校验失败: ' + error.message);
    }
}

// 更新指定地图的边宽度
function updateEdgeWidths(mapIndex) {
    // 如果未指定地图索引，则使用当前显示的地图
    const index = (mapIndex !== undefined) ? mapIndex : currentSlideIndex;
    
    // 获取指定地图的所有边
    const allEdges = edgeDatasets[index].get();
    
    if (allEdges.length === 0) {
        return; // 没有边，不需要更新
    }
    
    // 找出最大和最小值
    let minValue = Infinity;
    let maxValue = -Infinity;
    
    allEdges.forEach(edge => {
        const value = parseFloat(edge.value) || 1;
        if (value < minValue) minValue = value;
        if (value > maxValue) maxValue = value;
    });
    
    // 如果最大值等于最小值，所有边使用相同宽度
    if (maxValue === minValue) {
        allEdges.forEach(edge => {
            edgeDatasets[index].update({
                id: edge.id,
                width: 5 // 中间宽度
            });
        });
        
        // 只重新绘制指定地图
        if (networks[index]) {
            networks[index].redraw();
        }
        
        return;
    }
    
    // 设置边的宽度范围
    const minWidth = 1;  // 最小宽度
    const maxWidth = 10; // 最大宽度
    
    // 更新每条边的宽度
    allEdges.forEach(edge => {
        const value = parseFloat(edge.value) || 1;
        
        // 计算宽度（线性映射）
        const width = minWidth + ((value - minValue) / (maxValue - minValue)) * (maxWidth - minWidth);
        
        // Update edge width
        edgeDatasets[index].update({
            id: edge.id,
            width: width
        });
    });
    
    // 只重新绘制指定地图
    if (networks[index]) {
        networks[index].redraw();
    }
}

// Help模态框功能（支持中/英切换）
async function showHelpModal() {
    const modal = document.getElementById('help-modal');
    const helpContent = document.getElementById('help-content');
    const contentWrap = document.querySelector('#help-modal .modal-content');

    // 动态创建语言切换按钮（仅创建一次）
    if (!document.getElementById('help-lang-switch')) {
        const switchBar = document.createElement('div');
        switchBar.id = 'help-lang-switch';
        switchBar.style.display = 'flex';
        switchBar.style.justifyContent = 'flex-start';
        // 小尺寸样式
        const btnZh = document.createElement('button');
        btnZh.id = 'help-lang-zh';
        btnZh.textContent = '中文';
        btnZh.style.fontSize = '12px';
        btnZh.style.padding = '4px 8px';
        btnZh.style.lineHeight = '1';
        btnZh.style.borderRadius = '4px';
        const btnEn = document.createElement('button');
        btnEn.id = 'help-lang-en';
        btnEn.textContent = 'English';
        btnEn.style.fontSize = '12px';
        btnEn.style.padding = '4px 8px';
        btnEn.style.lineHeight = '1';
        btnEn.style.borderRadius = '4px';

        switchBar.appendChild(btnZh);
        switchBar.appendChild(btnEn);
        contentWrap.insertBefore(switchBar, helpContent);

        btnZh.addEventListener('click', async () => {
            localStorage.setItem('helpLanguage', 'zh');
            await renderHelp();
        });
        btnEn.addEventListener('click', async () => {
            localStorage.setItem('helpLanguage', 'en');
            await renderHelp();
        });
    }

    async function renderHelp() {
        const lang = localStorage.getItem('helpLanguage') || (navigator.language && navigator.language.startsWith('zh') ? 'zh' : 'en');
        try {
            const pathBase = lang === 'zh' ? 'front/chinese_description.md' : 'front/english_description.md';
            const response = await fetch(`${pathBase}?t=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`Failed to load help file: ${response.status}`);
            }
            const markdownText = await response.text();

            // 使用原有的 markdown 转 HTML
            const htmlContent = convertMarkdownToHTML(markdownText);
            helpContent.innerHTML = htmlContent;

            // 按钮状态反馈
            const btnZh = document.getElementById('help-lang-zh');
            const btnEn = document.getElementById('help-lang-en');
            if (btnZh && btnEn) {
                btnZh.disabled = (lang === 'zh');
                btnEn.disabled = (lang === 'en');
            }
        } catch (error) {
            console.error('加载帮助文件失败:', error);
            helpContent.innerHTML = '<p>无法加载帮助文档，请稍后重试。</p>';
        }
    }

    // 初次渲染并显示模态框
    await renderHelp();
    modal.style.display = 'block';
}

// 改进的markdown转HTML函数
function convertMarkdownToHTML(markdown) {
    // 按行分割处理
    const lines = markdown.split('\n');
    const result = [];
    let inList = false;
    let currentParagraph = [];
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        
        // 空行处理
        if (line === '') {
            if (inList) {
                result.push('</ul>');
                inList = false;
            }
            if (currentParagraph.length > 0) {
                result.push('<p>' + currentParagraph.join('<br>') + '</p>');
                currentParagraph = [];
            }
            continue;
        }
        
        // 标题处理
        if (line.match(/^#{1,5} /)) {
            if (inList) {
                result.push('</ul>');
                inList = false;
            }
            if (currentParagraph.length > 0) {
                result.push('<p>' + currentParagraph.join('<br>') + '</p>');
                currentParagraph = [];
            }
            
            const level = line.match(/^#+/)[0].length;
            const text = line.replace(/^#+\s*/, '');
            result.push(`<h${level}>${text}</h${level}>`);
            continue;
        }
        
        // 列表项处理
        if (line.match(/^- /)) {
            if (currentParagraph.length > 0) {
                result.push('<p>' + currentParagraph.join('<br>') + '</p>');
                currentParagraph = [];
            }
            if (!inList) {
                result.push('<ul>');
                inList = true;
            }
            const text = line.replace(/^- /, '').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            result.push(`<li>${text}</li>`);
            continue;
        }
        
        // 普通文本处理
        if (inList) {
            result.push('</ul>');
            inList = false;
        }
        
        // 处理粗体
        const processedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        currentParagraph.push(processedLine);
    }
    
    // 处理最后的内容
    if (inList) {
        result.push('</ul>');
    }
    if (currentParagraph.length > 0) {
        result.push('<p>' + currentParagraph.join('<br>') + '</p>');
    }
    
    return result.join('\n');
}

// ========== Evolution Mode ==========

function parseSheet4ForEvolution(sheet4Rows, sheet3Rows) {
    if (!sheet4Rows || sheet4Rows.length < 2) return null;

    const header = sheet4Rows[0].map(c => String(c === null || c === undefined ? '' : c).trim());
    const periodIdx = header.indexOf('period');
    const formsIdx = header.indexOf('forms');
    const languagesIdx = header.indexOf('languages');
    if (periodIdx < 0 || formsIdx < 0) return null;

    const funcCols = [];
    for (let c = 0; c < header.length; c++) {
        if (c === periodIdx || c === formsIdx || c === languagesIdx) continue;
        if (header[c].toLowerCase() === 'type') continue;
        if (header[c]) funcCols.push({ idx: c, name: header[c] });
    }

    const periodOrder = [];
    const periodSeen = new Set();
    const rows = [];
    let currentPeriod = '';
    for (let r = 1; r < sheet4Rows.length; r++) {
        const row = sheet4Rows[r];
        if (!row) continue;
        const pVal = row[periodIdx];
        if (pVal !== null && pVal !== undefined && String(pVal).trim()) {
            currentPeriod = String(pVal).trim();
        }
        const formVal = row[formsIdx];
        if (formVal === null || formVal === undefined || !String(formVal).trim()) continue;
        const form = String(formVal).trim();
        const lang = languagesIdx >= 0 && row[languagesIdx] ? String(row[languagesIdx]).trim() : '';

        if (!periodSeen.has(currentPeriod)) {
            periodOrder.push(currentPeriod);
            periodSeen.add(currentPeriod);
        }

        const functions = [];
        funcCols.forEach(fc => {
            const v = row[fc.idx];
            if (v === 1 || v === 1.0 || v === '1' || v === '1.0') {
                functions.push(fc.name);
            }
        });
        rows.push({ period: currentPeriod, form, language: lang, functions });
    }

    const words = [];
    const wordSeen = new Set();
    rows.forEach(r => {
        if (!wordSeen.has(r.form)) {
            words.push(r.form);
            wordSeen.add(r.form);
        }
    });

    const featureLabels = {};
    if (sheet3Rows && sheet3Rows.length >= 2) {
        const s3 = sheet3Rows;
        for (let c = 1; c < s3[0].length; c++) {
            const cn = s3[0] && s3[0][c] ? String(s3[0][c]).trim() : '';
            const en = s3[1] && s3[1][c] ? String(s3[1][c]).trim() : '';
            const abbr = s3[2] && s3[2][c] ? String(s3[2][c]).trim() : '';
            if (cn) featureLabels[cn] = { chinese: cn, english: en, abbrev: abbr };
            if (abbr && abbr !== cn) featureLabels[abbr] = { chinese: cn, english: en, abbrev: abbr };
        }
    }

    const data = {};
    words.forEach(w => {
        data[w] = {};
        periodOrder.forEach(p => {
            const fns = new Set();
            rows.filter(r => r.form === w && r.period === p).forEach(r => {
                r.functions.forEach(fn => fns.add(fn));
            });
            data[w][p] = { functions: Array.from(fns).sort() };
        });
    });

    return {
        dynasties: periodOrder,
        words: words,
        data: data,
        feature_labels: featureLabels
    };
}

async function enterEvolutionMode() {
    if (isEvolutionMode) {
        exitEvolutionMode();
        return;
    }

    if (!originalExcelData || !originalExcelData.data) {
        alert('Please load a dataset first.');
        return;
    }

    const sheet4Data = originalExcelData.sheet4;
    if (!sheet4Data || sheet4Data.length < 2) {
        alert('This dataset does not contain Sheet4 (diachronic evolution data). Evolution mode is not available.');
        return;
    }

    evolutionData = parseSheet4ForEvolution(sheet4Data, originalExcelData.sheet3);
    if (!evolutionData || !evolutionData.words || evolutionData.words.length === 0) {
        alert('Failed to parse Sheet4 evolution data. Please check the format.');
        return;
    }

    if (nodeDatasets.length > 0 && nodeDatasets[0]) {
        buildFeatureToNodeIdMap();
    } else {
        alert('Please wait for the graph to load and try again.');
        return;
    }

    isEvolutionMode = true;
    document.getElementById('evolution-panel').style.display = 'block';
    document.getElementById('filter-sections').style.display = 'none';
    document.getElementById('evolution-btn').textContent = 'Evolution ✕';
    document.getElementById('evolution-btn').classList.add('active');

    createEvolutionWordTabs();
    initEvolutionSlider();

    selectedEvolutionWords = [0];
    selectedEvolutionDynasty = 0;
    const slider = document.getElementById('evolution-slider');
    if (slider) slider.value = 0;
    updateEvolutionWordTabStyles();
    updateDynastyDisplay();
    updateEvolutionHighlight();
}

function exitEvolutionMode() {
    exitEvolutionModeInternal();
    highlightFormNodes('all');
    networks.forEach(n => { if (n) n.redraw(); });
}

function exitEvolutionModeInternal() {
    isEvolutionMode = false;
    document.getElementById('evolution-panel').style.display = 'none';
    document.getElementById('filter-sections').style.display = 'flex';
    document.getElementById('evolution-btn').textContent = 'Evolution';
    document.getElementById('evolution-btn').classList.remove('active');
    clearNodeHighlight();
    document.getElementById('evolution-verb-box').style.display = 'none';
    document.getElementById('evolution-corpus-box').style.display = 'none';
}

function buildFeatureToNodeIdMap() {
    evolutionFeatureToNodeId = {};
    if (nodeDatasets.length === 0) return;
    const nodes = nodeDatasets[0].get();
    nodes.forEach(node => {
        if (node.originalLabel) {
            evolutionFeatureToNodeId[node.originalLabel] = String(node.id);
            if (node.originalLabel.includes('/')) {
                node.originalLabel.split('/').forEach(part => {
                    evolutionFeatureToNodeId[part.trim()] = String(node.id);
                });
            }
        }
        if (node.label) {
            if (!evolutionFeatureToNodeId[node.label]) {
                evolutionFeatureToNodeId[node.label] = String(node.id);
            }
            if (node.label.includes('/')) {
                node.label.split('/').forEach(part => {
                    if (!evolutionFeatureToNodeId[part.trim()]) {
                        evolutionFeatureToNodeId[part.trim()] = String(node.id);
                    }
                });
            }
        }
        if (evolutionData && evolutionData.feature_labels) {
            const labels = evolutionData.feature_labels;
            const labelKeys = [node.originalLabel, node.label];
            labelKeys.forEach(key => {
                if (key && labels[key]) {
                    const info = labels[key];
                    if (info.abbrev && !evolutionFeatureToNodeId[info.abbrev]) {
                        evolutionFeatureToNodeId[info.abbrev] = String(node.id);
                    }
                    if (info.chinese && !evolutionFeatureToNodeId[info.chinese]) {
                        evolutionFeatureToNodeId[info.chinese] = String(node.id);
                    }
                    if (info.english && !evolutionFeatureToNodeId[info.english]) {
                        evolutionFeatureToNodeId[info.english] = String(node.id);
                    }
                }
            });
        }
    });
}

function createEvolutionWordTabs() {
    const tabs = document.getElementById('evolution-word-tabs');
    if (!tabs || !evolutionData) return;
    tabs.innerHTML = '';

    evolutionData.words.forEach((word, idx) => {
        const tab = document.createElement('button');
        tab.className = 'evolution-word-tab';
        tab.textContent = word;
        tab.dataset.index = idx;
        tab.addEventListener('click', () => toggleEvolutionWord(idx));
        tabs.appendChild(tab);
    });
}

function toggleEvolutionWord(idx) {
    const wordIdx = selectedEvolutionWords.indexOf(idx);
    if (wordIdx >= 0) {
        selectedEvolutionWords.splice(wordIdx, 1);
    } else {
        selectedEvolutionWords.push(idx);
    }
    updateEvolutionWordTabStyles();
    updateEvolutionHighlight();
}

function updateEvolutionWordTabStyles() {
    const tabs = document.querySelectorAll('.evolution-word-tab');
    tabs.forEach((tab, i) => {
        if (selectedEvolutionWords.includes(i)) {
            tab.classList.add('selected');
            tab.style.borderColor = FORM_COLORS[i % FORM_COLORS.length];
            tab.style.backgroundColor = FORM_COLORS[i % FORM_COLORS.length] + '1a';
            tab.style.color = FORM_COLORS[i % FORM_COLORS.length];
        } else {
            tab.classList.remove('selected');
            tab.style.borderColor = '';
            tab.style.backgroundColor = '';
            tab.style.color = '';
        }
    });
}

function initEvolutionSlider() {
    const slider = document.getElementById('evolution-slider');
    const labelsContainer = document.getElementById('evolution-slider-labels');
    if (!slider || !labelsContainer || !evolutionData) return;

    slider.max = evolutionData.dynasties.length - 1;
    slider.value = 0;
    selectedEvolutionDynasty = 0;

    labelsContainer.innerHTML = '';
    evolutionData.dynasties.forEach((dynasty, idx) => {
        const label = document.createElement('span');
        label.className = 'evolution-slider-label';
        label.textContent = dynasty;
        label.addEventListener('click', () => {
            slider.value = idx;
            selectedEvolutionDynasty = idx;
            updateDynastyDisplay();
            updateEvolutionHighlight();
        });
        labelsContainer.appendChild(label);
    });

    slider.oninput = function() {
        selectedEvolutionDynasty = parseInt(this.value);
        updateDynastyDisplay();
        updateEvolutionHighlight();
    };

    updateDynastyDisplay();
}

function updateDynastyDisplay() {
    const display = document.getElementById('evolution-dynasty-display');
    if (display && evolutionData && evolutionData.dynasties[selectedEvolutionDynasty]) {
        display.textContent = evolutionData.dynasties[selectedEvolutionDynasty];
    }

    const labels = document.querySelectorAll('.evolution-slider-label');
    labels.forEach((label, idx) => {
        if (idx === selectedEvolutionDynasty) {
            label.classList.add('active');
        } else {
            label.classList.remove('active');
        }
    });
}

function updateEvolutionHighlight() {
    clearNodeHighlight();

    if (!evolutionData || selectedEvolutionWords.length === 0) {
        networks.forEach(n => { if (n) n.redraw(); });
        document.getElementById('evolution-verb-box').style.display = 'none';
        document.getElementById('evolution-corpus-box').style.display = 'none';
        document.getElementById('form-legend').style.display = 'none';
        return;
    }

    const dynasty = evolutionData.dynasties[selectedEvolutionDynasty];
    const legend = document.getElementById('form-legend');
    legend.innerHTML = '';

    selectedEvolutionWords.forEach(wordIdx => {
        const word = evolutionData.words[wordIdx];
        const wordData = evolutionData.data[word];
        if (!wordData) return;

        const dynastyData = wordData[dynasty];
        const color = FORM_COLORS[wordIdx % FORM_COLORS.length];
        const nodeIds = [];

        if (dynastyData && dynastyData.functions) {
            dynastyData.functions.forEach(featureName => {
                const nodeId = evolutionFeatureToNodeId[featureName];
                if (nodeId !== undefined && !nodeIds.includes(nodeId)) {
                    nodeIds.push(nodeId);
                }
            });
        }

        if (nodeIds.length > 0) {
            currentFormOutlines.push({
                formName: word + ' (' + dynasty + ')',
                color: color,
                nodeIds: nodeIds,
                isOverlay: false
            });
            nodeIds.forEach(nid => {
                if (!highlightedNodes.includes(nid)) {
                    highlightedNodes.push(nid);
                }
            });
        }

        const item = document.createElement('div');
        item.className = 'form-legend-item';
        item.style.borderColor = color;
        const dot = document.createElement('span');
        dot.className = 'form-legend-dot';
        dot.style.background = color;
        const labelEl = document.createElement('span');
        const funcCount = dynastyData ? dynastyData.functions.length : 0;
        labelEl.textContent = word + ' (' + funcCount + ' functions)';
        item.appendChild(dot);
        item.appendChild(labelEl);
        legend.appendChild(item);
    });

    legend.style.display = 'flex';
    updateVerbSemanticsBox(dynasty);
    if (evolutionData.corpus) {
        updateEvolutionCorpusPanel(dynasty);
    } else {
        document.getElementById('evolution-corpus-box').style.display = 'none';
    }

    networks.forEach(n => { if (n) n.redraw(); });
    setTimeout(updateComparisonPanel, 100);
}

function escapeRegExp(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function highlightWordInText(text, word) {
    const escaped = escapeRegExp(word);
    const re = new RegExp(escaped, 'g');
    return text.replace(re, '<b>$&</b>');
}

// Resolve a hovered node's label(s) to a corpus feature key (Chinese name)
function resolveCorpusFeatureKey(nodeData) {
    if (!nodeData || !evolutionData || !evolutionData.corpus) return null;
    const seen = new Set();
    const candidates = [];
    function push(v) { const s = String(v).trim(); if (s && !seen.has(s)) { seen.add(s); candidates.push(s); } }
    if (nodeData.label) push(nodeData.label);
    if (nodeData.originalLabel) push(nodeData.originalLabel);
    // Expand slash-separated labels on a snapshot
    [...candidates].forEach(l => {
        if (l.includes('/')) l.split('/').forEach(p => { if (p.trim()) push(p.trim()); });
    });
    // Add chinese / abbrev aliases on a snapshot
    [...candidates].forEach(l => {
        const info = supplementaryFeatureLabels[l];
        if (!info) return;
        if (info.chinese) push(info.chinese);
        if (info.abbrev) push(info.abbrev);
    });
    const dynasty = evolutionData.dynasties[selectedEvolutionDynasty];
    for (const w of evolutionData.words) {
        const pc = evolutionData.corpus[w] ? evolutionData.corpus[w][dynasty] : null;
        if (!pc) continue;
        for (const l of candidates) {
            if (pc[l]) return l;
        }
    }
    return null;
}

// Build hover tooltip HTML aggregating corpus examples for a feature node
function buildEvolutionNodeTooltip(nodeData) {
    if (!evolutionData || !evolutionData.corpus) return '';
    const dynasty = evolutionData.dynasties[selectedEvolutionDynasty];
    const featureKey = resolveCorpusFeatureKey(nodeData);
    if (!featureKey) return '';

    // Collect matching entries from each selected word
    const blocks = [];
    selectedEvolutionWords.forEach(wordIdx => {
        const word = evolutionData.words[wordIdx];
        const periodCorpus = evolutionData.corpus[word] ? evolutionData.corpus[word][dynasty] : null;
        if (!periodCorpus) return;
        const entries = periodCorpus[featureKey];
        if (!entries || entries.length === 0) return;
        const color = FORM_COLORS[wordIdx % FORM_COLORS.length];
        const flInfo = evolutionData.feature_labels ? evolutionData.feature_labels[featureKey] : null;
        const enName = flInfo && flInfo.english ? flInfo.english : '';
        let html = '<div style="margin-bottom:8px;border-left:3px solid ' + color + ';padding-left:8px;">';
        html += '<div style="font-weight:700;color:' + color + ';font-size:13px;margin-bottom:3px;">'
              + word + (enName ? ' · ' + featureKey + ' / ' + enName : ' · ' + (featureKey || ''))
              + ' (' + dynasty + ')</div>';
        entries.forEach(e => {
            const dynLabel = e.dynasty ? e.dynasty.replace(/^\d+/, '') : '';
            const txt = e.text.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\n/g, '<br>');
            html += '<div style="font-size:12px;line-height:1.6;color:#333;margin:2px 0;">';
            if (dynLabel) html += '<span style="display:inline-block;font-size:10px;color:#888;background:#f0f0f0;padding:0 5px;border-radius:7px;margin-right:5px;">' + dynLabel + '</span>';
            html += highlightWordInText(txt, word) + '</div>';
        });
        html += '</div>';
        blocks.push(html);
    });

    if (blocks.length === 0) return '';
    return blocks.join('');
}

function updateEvolutionCorpusPanel(dynasty) {
    const box = document.getElementById('evolution-corpus-box');
    const content = document.getElementById('evolution-corpus-content');
    if (!box || !content || !evolutionData) return;

    if (selectedEvolutionWords.length === 0) {
        box.style.display = 'none';
        return;
    }

    content.innerHTML = '';
    let hasAny = false;

    selectedEvolutionWords.forEach(wordIdx => {
        const word = evolutionData.words[wordIdx];
        const wordCorpus = evolutionData.corpus ? evolutionData.corpus[word] : null;
        if (!wordCorpus) return;

        const periodCorpus = wordCorpus[dynasty];
        if (!periodCorpus || Object.keys(periodCorpus).length === 0) return;

        hasAny = true;
        const color = FORM_COLORS[wordIdx % FORM_COLORS.length];

        const block = document.createElement('div');
        block.className = 'evolution-corpus-word-block';
        block.style.borderLeftColor = color;

        const head = document.createElement('div');
        head.className = 'evolution-corpus-word-head';
        head.style.color = color;
        head.textContent = word + ' (' + dynasty + ')';
        block.appendChild(head);

        // Order: repetitive-meaning (重复) first, then the rest sorted
        const funcNames = Object.keys(periodCorpus);
        const ordered = funcNames.includes('重复')
            ? ['重复', ...funcNames.filter(f => f !== '重复')]
            : funcNames.slice().sort();

        ordered.forEach(funcName => {
            const entries = periodCorpus[funcName];
            const funcDiv = document.createElement('div');
            funcDiv.className = 'evolution-corpus-func';

            const label = document.createElement('span');
            label.className = 'evolution-corpus-func-label';
            const flInfo = evolutionData.feature_labels ? evolutionData.feature_labels[funcName] : null;
            const enName = flInfo && flInfo.english ? flInfo.english : funcName;
            label.textContent = funcName + (flInfo && flInfo.english ? ' / ' + flInfo.english : '');
            label.title = enName;
            funcDiv.appendChild(label);

            entries.forEach(entry => {
                const entryDiv = document.createElement('div');
                entryDiv.className = 'evolution-corpus-entry';
                if (entry.dynasty) {
                    const dyn = document.createElement('span');
                    dyn.className = 'evolution-corpus-dynasty';
                    dyn.textContent = entry.dynasty.replace(/^\d+/, '');
                    entryDiv.appendChild(dyn);
                }
                const txt = document.createElement('span');
                txt.className = 'evolution-corpus-text';
                const withBreaks = entry.text.replace(/\n/g, '<br>');
                txt.innerHTML = highlightWordInText(withBreaks, word);
                entryDiv.appendChild(txt);
                funcDiv.appendChild(entryDiv);
            });

            block.appendChild(funcDiv);
        });

        content.appendChild(block);
    });

    if (!hasAny) {
        const empty = document.createElement('div');
        empty.className = 'evolution-corpus-empty';
        empty.textContent = 'No corpus examples for the selected word(s) in this period.';
        content.appendChild(empty);
    }

    box.style.display = 'block';
}

function updateVerbSemanticsBox(dynasty) {
    const verbBox = document.getElementById('evolution-verb-box');
    const verbContent = document.getElementById('evolution-verb-content');
    if (!verbBox || !verbContent || !evolutionData) return;

    if (!evolutionData.verb_features || evolutionData.verb_features.length === 0) {
        verbBox.style.display = 'none';
        return;
    }

    let hasVerbs = false;
    verbContent.innerHTML = '';

    selectedEvolutionWords.forEach(wordIdx => {
        const word = evolutionData.words[wordIdx];
        const wordData = evolutionData.data[word];
        if (!wordData) return;

        const dynastyData = wordData[dynasty];
        if (!dynastyData || !dynastyData.verbs || dynastyData.verbs.length === 0) return;

        hasVerbs = true;
        const color = FORM_COLORS[wordIdx % FORM_COLORS.length];

        const wordDiv = document.createElement('div');
        wordDiv.className = 'evolution-verb-word';

        const wordLabel = document.createElement('span');
        wordLabel.className = 'evolution-verb-word-label';
        wordLabel.style.color = color;
        wordLabel.textContent = word + ':';
        wordDiv.appendChild(wordLabel);

        dynastyData.verbs.forEach(verb => {
            const tag = document.createElement('span');
            tag.className = 'evolution-verb-tag';
            const verbInfo = evolutionData.verb_labels ? evolutionData.verb_labels[verb] : null;
            if (verbInfo) {
                tag.textContent = verbInfo.chinese + ' (' + verbInfo.english + ')';
                tag.title = verb;
            } else {
                tag.textContent = verb.replace('_动', '(动)').replace('_数', '(数)');
            }
            wordDiv.appendChild(tag);
        });

        verbContent.appendChild(wordDiv);
    });

    verbBox.style.display = hasVerbs ? 'block' : 'none';
}

// ========== SLA Mode ==========

function enterSLAMode() {
    if (isSLAMode) {
        exitSLAMode();
        return;
    }

    if (!originalExcelData || !originalExcelData.data) {
        alert('Please load a dataset first.');
        return;
    }

    const sheet5Data = originalExcelData.sheet5;
    const hasSheet5 = sheet5Data && sheet5Data.length >= 2;

    // Extract languages and forms from Sheet1 data
    const sheet1Rows = originalExcelData.data;
    const languagesSet = new Set();
    const formsSet = new Set();
    sheet1Rows.forEach(row => {
        if (row && row.languages) languagesSet.add(String(row.languages).trim());
        if (row && row.forms) formsSet.add(String(row.forms).trim());
    });
    slaSheet1Languages = Array.from(languagesSet);
    slaSheet1Forms = Array.from(formsSet);

    isSLAMode = true;
    document.getElementById('sla-panel').style.display = 'block';
    document.getElementById('filter-sections').style.display = 'none';
    document.getElementById('sla-btn').textContent = 'SLA ✕';
    document.getElementById('sla-btn').classList.add('active');

    // Populate language datalists
    const nativeList = document.getElementById('sla-native-lang-list');
    const targetList = document.getElementById('sla-target-lang-list');
    nativeList.innerHTML = '';
    targetList.innerHTML = '';
    slaSheet1Languages.forEach(lang => {
        const o1 = document.createElement('option'); o1.value = lang; nativeList.appendChild(o1);
        const o2 = document.createElement('option'); o2.value = lang; targetList.appendChild(o2);
    });

    // Clear input values
    document.getElementById('sla-native-lang').value = '';
    document.getElementById('sla-target-lang').value = '';

    // Populate form dropdown
    const formSel = document.getElementById('sla-form-select');
    formSel.innerHTML = '<option value="">-- Select --</option>';
    slaSheet1Forms.forEach(f => formSel.appendChild(new Option(f, f)));

    // Level selector: disable if no Sheet5
    const levelSel = document.getElementById('sla-level-select');
    if (hasSheet5) {
        levelSel.disabled = false;
        levelSel.value = '初级';
    } else {
        levelSel.disabled = true;
        levelSel.innerHTML = '<option value="">N/A</option>';
    }

    // Clear result info
    document.getElementById('sla-result-info').style.display = 'none';

    // Auto-generate on form or language change; level change draws outlines only
    const nativeInput = document.getElementById('sla-native-lang');
    const targetInput = document.getElementById('sla-target-lang');
    formSel.onchange = autoGenerateSLAGraph;
    nativeInput.onchange = autoGenerateSLAGraph;
    targetInput.onchange = autoGenerateSLAGraph;
    levelSel.onchange = hasSheet5 ? drawSLALevelOutlines : null;

    // Save current graph for restore on exit
    slaSavedGraphData = {
        graphData: null,
        titles: null
    };
    // nodeDatasets holds the current graph nodes
    if (nodeDatasets && nodeDatasets.length > 0) {
        slaSavedGraphData.graphData = nodeDatasets.map(ds => ds ? ds.get() : []);
    }
    // Clear level state
    slaSheet5LevelFeatures = null;
}

function exitSLAMode() {
    exitSLAModeInternal();
    // Reload the original graph
    if (originalExcelData && originalExcelData.data) {
        refetchGraph();
    }
}

function exitSLAModeInternal() {
    isSLAMode = false;
    slaLevelOutlines = [];
    document.getElementById('sla-panel').style.display = 'none';
    document.getElementById('filter-sections').style.display = 'flex';
    document.getElementById('sla-btn').textContent = 'SLA';
    document.getElementById('sla-btn').classList.remove('active');
    document.getElementById('sla-result-info').style.display = 'none';
    // Clean up auto-generate listeners
    const formSel = document.getElementById('sla-form-select');
    const levelSel = document.getElementById('sla-level-select');
    const nativeInput = document.getElementById('sla-native-lang');
    const targetInput = document.getElementById('sla-target-lang');
    if (formSel) formSel.onchange = null;
    if (levelSel) levelSel.onchange = null;
    if (nativeInput) nativeInput.onchange = null;
    if (targetInput) targetInput.onchange = null;
}

function autoGenerateSLAGraph() {
    const form = document.getElementById('sla-form-select').value;
    if (!form) return; // need a form selected
    generateSLAGraph();
}

async function generateSLAGraph() {
    const nativeLang = document.getElementById('sla-native-lang').value.trim();
    const targetLang = document.getElementById('sla-target-lang').value.trim();
    const form = document.getElementById('sla-form-select').value;
    const level = document.getElementById('sla-level-select').value;
    const sheet5Rows = originalExcelData.sheet5;
    const hasSheet5 = sheet5Rows && sheet5Rows.length >= 2;

    if (!form) return;
    if (hasSheet5 && !level) return;

    const resultInfo = document.getElementById('sla-result-info');
    resultInfo.style.display = 'block';
    resultInfo.textContent = 'Generating concept space...';

    try {
        let activeFeatures = [];
        if (hasSheet5) {
            // Parse Sheet5 for ALL levels (store for outline drawing)
            const header = sheet5Rows[0].map(c => String(c === null || c === undefined ? '' : c).trim());
            const levelIdx = header.indexOf('level');
            if (levelIdx < 0) {
                resultInfo.textContent = 'Error: Sheet5 does not have a "level" column.';
                return;
            }

            const funcCols = [];
            for (let c = 0; c < header.length; c++) {
                if (c === levelIdx) continue;
                if (header[c]) funcCols.push({ idx: c, name: header[c] });
            }

            // Build level→features map for all levels
            const levelFeaturesMap = {};
            for (let r = 1; r < sheet5Rows.length; r++) {
                const row = sheet5Rows[r];
                if (!row) continue;
                const lvl = String(row[levelIdx]).trim();
                const feats = [];
                funcCols.forEach(fc => {
                    const v = row[fc.idx];
                    if (v === 1 || v === 1.0 || v === '1' || v === '1.0') {
                        feats.push(fc.name);
                    }
                });
                levelFeaturesMap[lvl] = feats;
            }

            activeFeatures = levelFeaturesMap[level] || [];
            if (activeFeatures.length === 0) {
                resultInfo.textContent = 'No active features found for level "' + level + '".';
                return;
            }

            // Store for outline drawing
            slaSheet5LevelFeatures = levelFeaturesMap;
        } else {
            slaSheet5LevelFeatures = null;
        }

        // Always send ALL Sheet1 rows to backend (full graph)
        const sheet1Rows = originalExcelData.data;

        // Send to backend for concept space computation
        const blendSettings = getCurrentBlendSettings();
        const response = await fetch('/api/sla-concept-space', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sheet1_data: sheet1Rows,
                sheet5_features: activeFeatures,
                form: form,
                native_lang: nativeLang || '',
                target_lang: targetLang || '',
                level: level,
                calc_type: blendSettings.calcType,
                blend_weight: blendSettings.blendWeight
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `HTTP ${response.status}`);
        }

        const result = await response.json();

        // Render the result graph
        if (result.graph_data && result.graph_data.length > 0) {
            const titleLevel = level || 'no level';
            updateGraphTitles('SLA: ' + form + ' (' + titleLevel + ')', result.graph_data);
            window.currentFormsData = result.forms_with_nodes || [];
            updateFormFilter(result.forms_with_nodes || [], {});

            // Auto-draw outlines only for forms matching native/target languages
            if (window.currentFormsData.length > 0) {
                const nLang = document.getElementById('sla-native-lang').value.trim();
                const tLang = document.getElementById('sla-target-lang').value.trim();
                const matchingIndices = [];
                window.currentFormsData.forEach((fd, i) => {
                    const lang = String(fd.language || '').split(' (')[0].trim();
                    if (!nLang && !tLang) {
                        matchingIndices.push(i); // no filter, show all
                    } else if ((nLang && lang === nLang) || (tLang && lang === tLang)) {
                        matchingIndices.push(i);
                    }
                });
                setTimeout(() => {
                    if (matchingIndices.length > 0) {
                        highlightFormNodes(matchingIndices);
                    } else {
                        clearNodeHighlight();
                    }
                    if (hasSheet5) drawSLALevelOutlines();
                }, 200);
            }
        }

        // Show result info
        const nativeDisplay = nativeLang || '(all)';
        const targetDisplay = targetLang || '(all)';
        let matchInfo = '';
        if (nativeLang && targetLang) {
            const matchCount = window.currentFormsData.filter(fd => {
                const lang = String(fd.language || '').split(' (')[0].trim();
                return lang === nativeLang || lang === targetLang;
            }).length;
            matchInfo = ' | <b>Matched forms:</b> ' + matchCount;
        }
        const levelInfo = level ? 
            ' | <b>Level:</b> ' + level + ' | <b>Active features:</b> ' + activeFeatures.length : '';
        resultInfo.innerHTML = '<b>Concept Space for:</b> ' + form + 
            ' | <b>Native:</b> ' + nativeDisplay + 
            ' | <b>Target:</b> ' + targetDisplay + 
            levelInfo +
            ' | <b>Forms:</b> ' + (result.forms_with_nodes ? result.forms_with_nodes.length : 0) +
            matchInfo;

    } catch (error) {
        console.error('SLA generation error:', error);
        resultInfo.textContent = 'Error: ' + error.message;
    }
}

function drawSLALevelOutlines() {
    const level = document.getElementById('sla-level-select').value;
    if (!level || !slaSheet5LevelFeatures || !slaSheet5LevelFeatures[level]) {
        slaLevelOutlines = [];
        networks.forEach(n => { if (n) n.redraw(); });
        return;
    }

    const levelFeats = slaSheet5LevelFeatures[level];
    const allFeats = slaSheet5LevelFeatures;

    // Collect all possible feature names from Sheet5
    const allFeatureNames = new Set();
    Object.values(allFeats).forEach(feats => feats.forEach(f => allFeatureNames.add(f)));

    // Map feature names to node IDs from the current graph
    const featToNodeId = {};

    // Check all node datasets (graph slides) — they all have the same nodes
    nodeDatasets.forEach(ds => {
        if (!ds) return;
        ds.forEach(node => {
            const label = String(node.originalLabel || node.label || '').trim();
            const parts = label.split('/');
            parts.forEach(part => {
                const p = part.trim();
                if (allFeatureNames.has(p)) featToNodeId[p] = String(node.id);
            });
            if (allFeatureNames.has(label)) featToNodeId[label] = String(node.id);
        });
    });

    // Build outlines for each level — active level solid, others lighter
    const LEVEL_COLORS = {
        '初级': '#81C784',
        '中级': '#64B5F6',
        '高级': '#BA68C8'
    };

    slaLevelOutlines = [];
    // Draw all three levels so user can compare; active one is bold
    ['初级', '中级', '高级'].forEach(lvl => {
        const feats = allFeats[lvl];
        if (!feats || feats.length === 0) return;
        const nodeIds = feats.map(f => featToNodeId[f]).filter(id => id != null);
        if (nodeIds.length === 0) return;
        slaLevelOutlines.push({
            levelName: lvl,
            color: LEVEL_COLORS[lvl] || '#999',
            nodeIds: nodeIds,
            isActive: (lvl === level)
        });
    });

    // Update legend
    const legend = document.getElementById('form-legend');
    if (legend) {
        legend.innerHTML = '';
        slaLevelOutlines.forEach(ol => {
            const item = document.createElement('div');
            item.className = 'form-legend-item';
            item.style.borderColor = ol.color;
            const dot = document.createElement('span');
            dot.className = 'form-legend-dot';
            dot.style.background = ol.color;
            const label = document.createElement('span');
            label.textContent = ol.levelName + ' (' + ol.nodeIds.length + ')';
            if (!ol.isActive) label.style.opacity = '0.5';
            item.appendChild(dot);
            item.appendChild(label);
            legend.appendChild(item);
        });
        legend.style.display = 'flex';
    }

    networks.forEach(n => { if (n) n.redraw(); });
}

// 模态框关闭功能
document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('help-modal');
    const closeBtn = document.querySelector('.close');
    
    // 点击关闭按钮关闭模态框
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            modal.style.display = 'none';
        });
    }
    
    // 点击模态框外部关闭
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
});