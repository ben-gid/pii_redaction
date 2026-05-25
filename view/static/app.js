function navigate(offset) {
    const currentIdx = parseInt(document.getElementById('index-input').value);
    const nextIdx = currentIdx + offset;
    const split = document.getElementById('split-select').value;
    window.location.href = `/?split=${split}&index=${nextIdx}`;
}

function goRandom() {
    const split = document.getElementById('split-select').value;
    const totalExamples = parseInt(document.body.dataset.totalExamples);
    const randomIdx = Math.floor(Math.random() * totalExamples);
    window.location.href = `/?split=${split}&index=${randomIdx}`;
}

function switchTab(tab) {
    const highlightTab = document.getElementById('tab-highlight');
    const jsonTab = document.getElementById('tab-json');
    const highlightContent = document.getElementById('panel-highlight-content');
    const jsonContent = document.getElementById('panel-json-content');

    if (tab === 'highlight') {
        highlightTab.classList.add('active');
        jsonTab.classList.remove('active');
        highlightContent.style.display = 'block';
        jsonContent.style.display = 'none';
    } else {
        jsonTab.classList.add('active');
        highlightTab.classList.remove('active');
        jsonContent.style.display = 'block';
        highlightContent.style.display = 'none';
    }
}
