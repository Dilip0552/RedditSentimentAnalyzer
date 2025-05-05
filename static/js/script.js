document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const searchForm = document.getElementById('search-form');
    const datasetForm = document.getElementById('dataset-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const datasetAnalyzeBtn = document.getElementById('analyze-dataset-btn');
    const btnText = document.getElementById('btn-text');
    const datasetBtnText = document.getElementById('dataset-btn-text');
    const loadingSpinner = document.getElementById('loading-spinner');
    const datasetLoadingSpinner = document.getElementById('dataset-loading-spinner');
    const datasetSelect = document.getElementById('dataset-select');
    const filterTopic = document.getElementById('filter-topic');
    const filterSubreddit = document.getElementById('filter-subreddit');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const resultsContainer = document.getElementById('results-container');
    const emptyState = document.getElementById('empty-state');
    const sentimentTable = document.getElementById('sentiment-table');
    const positiveComments = document.getElementById('positive-comments');
    const negativeComments = document.getElementById('negative-comments');
    const neutralComments = document.getElementById('neutral-comments');
    const positiveCount = document.getElementById('positive-count');
    const negativeCount = document.getElementById('negative-count');
    const neutralCount = document.getElementById('neutral-count');
    const liveAnalysisSection = document.getElementById('live-analysis-section');
    const preanalyzedSection = document.getElementById('preanalyzed-section');
    const modeSelection = document.getElementById('mode-selection');
    
    // Get all mode cards
    const modeCards = document.querySelectorAll('.mode-card');
    
    // Chart objects
    let doughnutChart = null;
    let timelineChart = null;
    let emotionsChart = null;
    
    // Subreddit list for the filter
    let availableSubreddits = [];
    
    // Current selected mode
    let currentMode = null;
    
    // Event listeners
    searchForm.addEventListener('submit', analyzeReddit);
    datasetForm.addEventListener('submit', analyzeDataset);
    datasetSelect.addEventListener('change', updateSubredditOptions);
    
    // Add event listeners to mode cards
    modeCards.forEach(card => {
        card.addEventListener('click', function() {
            const mode = this.getAttribute('data-mode');
            selectMode(mode);
        });
    });
    
    // Load available datasets on page load
    loadAvailableDatasets();
    
    // Global function for selecting mode
    function selectMode(mode) {
        currentMode = mode;
        
        // Reset all cards first
        modeCards.forEach(card => {
            card.classList.remove('selected');
            const badge = card.querySelector('.mode-selected');
            if (badge) badge.classList.add('d-none');
        });
        
        // Update UI based on selected mode
        if (mode === 'live-analysis') {
            liveAnalysisSection.classList.remove('d-none');
            preanalyzedSection.classList.add('d-none');
            
            // Highlight selected card
            const selectedCard = document.querySelector(`.mode-card[data-mode="live-analysis"]`);
            selectedCard.classList.add('selected');
            const badge = selectedCard.querySelector('.mode-selected');
            if (badge) badge.classList.remove('d-none');
            
        } else if (mode === 'preanalyzed') {
            liveAnalysisSection.classList.add('d-none');
            preanalyzedSection.classList.remove('d-none');
            
            // Highlight selected card
            const selectedCard = document.querySelector(`.mode-card[data-mode="preanalyzed"]`);
            selectedCard.classList.add('selected');
            const badge = selectedCard.querySelector('.mode-selected');
            if (badge) badge.classList.remove('d-none');
        }
        
        // Hide results and errors when switching modes
        resultsContainer.classList.add('d-none');
        emptyState.classList.remove('d-none');
        hideError();
    };
    
    // Function to analyze Reddit comments
    function analyzeReddit(e) {
        e.preventDefault();
        
        // Get form values
        const topic = document.getElementById('topic').value.trim();
        const subreddit = document.getElementById('subreddit').value.trim() || 'all';
        const includeEmotion = document.getElementById('include-emotion').checked;
        
        // Validate input
        if (!topic) {
            showError('Please enter a topic to analyze');
            return;
        }
        
        // Show loading state
        setLoadingState(true);
        hideError();
        
        // Make API request
        fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                topic, 
                subreddit,
                include_emotion: includeEmotion
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'An error occurred during analysis');
                });
            }
            return response.json();
        })
        .then(data => {
            // Process and display results
            displayResults(data, topic, subreddit);
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message || 'Failed to analyze data from Reddit');
        })
        .finally(() => {
            setLoadingState(false);
        });
    }
    
    // Function to set loading state
    function setLoadingState(isLoading) {
        if (isLoading) {
            btnText.textContent = 'Analyzing...';
            loadingSpinner.classList.remove('d-none');
            analyzeBtn.disabled = true;
        } else {
            btnText.textContent = 'Analyze';
            loadingSpinner.classList.add('d-none');
            analyzeBtn.disabled = false;
        }
    }
    
    // Function to show error message
    function showError(message) {
        errorText.textContent = message;
        errorMessage.classList.remove('d-none');
    }
    
    // Function to hide error message
    function hideError() {
        errorMessage.classList.add('d-none');
    }
    
    // Function to display analysis results
    function displayResults(data, topic, subreddit) {
        // Show results container and hide empty state
        resultsContainer.classList.remove('d-none');
        emptyState.classList.add('d-none');
        
        // Get emotions container
        const emotionsContainer = document.getElementById('emotions-container');
        
        // Update topic and subreddit display
        document.getElementById('topic-display').textContent = topic || 'N/A';
        document.getElementById('subreddit-display').textContent = subreddit || 'all';
        
        // Update total comments count
        const totalComments = data.positive.count + data.negative.count + data.neutral.count;
        document.getElementById('total-comments-count').textContent = totalComments;
        
        // Update time period if available (can be expanded based on data)
        const timePeriod = document.getElementById('time-period');
        if (data.time_range && data.time_range.start && data.time_range.end) {
            timePeriod.textContent = `${data.time_range.start} to ${data.time_range.end}`;
        } else {
            timePeriod.textContent = 'All time';
        }
        
        // Update sentiment volatility if available
        const sentimentVolatility = document.getElementById('sentiment-volatility');
        if (data.volatility) {
            sentimentVolatility.textContent = data.volatility;
        } else {
            sentimentVolatility.textContent = 'N/A';
        }
        
        // Update sentiment counts
        positiveCount.textContent = data.positive.count;
        negativeCount.textContent = data.negative.count;
        neutralCount.textContent = data.neutral.count;
        
        // Display sentiment table
        updateSentimentTable(data);
        
        // Display doughnut chart
        createDoughnutChart(data);
        
        // Display timeline chart
        if (data.by_month && data.by_month.length > 0) {
            // Use monthly data for more detailed temporal analysis if available
            createTimelineChart(data.by_month, 'month');
        } else {
            // Fall back to yearly data
            createTimelineChart(data.by_year, 'year');
        }
        
        // Display comment samples
        displayComments('positive', data.positive.samples, positiveComments);
        displayComments('negative', data.negative.samples, negativeComments);
        displayComments('neutral', data.neutral.samples, neutralComments);
        
        // Create and display sentiment summary
        createSentimentSummary(data, topic, subreddit);
        
        // Create and display statistical analysis
        createAnalysisSummary(data, topic, subreddit);
        
        // Display word frequency data
        if (data.word_freq) {
            createWordFrequencyDisplay('positive', data.word_freq.positive);
            createWordFrequencyDisplay('negative', data.word_freq.negative);
            createWordFrequencyDisplay('neutral', data.word_freq.neutral);
        }
        
        // Display emotions data if available
        if (data.emotions && Object.keys(data.emotions).length > 0) {
            emotionsContainer.classList.remove('d-none');
            createEmotionsChart(data.emotions);
        } else {
            emotionsContainer.classList.add('d-none');
        }
        
        // Display topics if available
        if (data.topics && (Array.isArray(data.topics) && data.topics.length > 0)) {
            const topicsContainer = document.getElementById('topics-container');
            const noTopicsMessage = document.getElementById('no-topics-message');
            
            // Clear previous topics
            const listGroup = topicsContainer.querySelector('.list-group');
            
            // Remove existing topics (but keep the no-topics message)
            Array.from(listGroup.children).forEach(child => {
                if (child.id !== 'no-topics-message') {
                    child.remove();
                }
            });
            
            // Hide the no topics message
            noTopicsMessage.classList.add('d-none');
            
            // Create and display topic items
            data.topics.forEach((topic, index) => {
                const topicElement = document.createElement('div');
                topicElement.className = 'list-group-item';
                
                const topicHeader = document.createElement('h6');
                topicHeader.innerHTML = `<i class="fas fa-hashtag me-2 text-info"></i>Topic ${index + 1}`;
                
                const topicKeywords = document.createElement('p');
                topicKeywords.className = 'mb-0 small';
                
                // Improved handling of topic formats with better logging
                let keywords = [];
                
                // Handle different possible topic formats
                if (Array.isArray(topic)) {
                    // Format 1: Direct array of strings
                    keywords = topic;
                    console.log(`Topic ${index + 1} is an array of strings:`, keywords);
                } else if (typeof topic === 'object' && topic !== null) {
                    // Format 2: Object with 'words' property (array)
                    if (topic.words && Array.isArray(topic.words)) {
                        keywords = topic.words;
                        console.log(`Topic ${index + 1} is an object with words property:`, keywords);
                    } 
                    // Format 3: Object with 'id' and 'words' properties
                    else if (topic.id !== undefined && topic.words && Array.isArray(topic.words)) {
                        keywords = topic.words;
                        console.log(`Topic ${index + 1} is an object with id and words properties:`, keywords);
                    }
                    // Format 4: Just handle it as entries
                    else {
                        try {
                            keywords = Object.entries(topic).map(([key, value]) => 
                                typeof value === 'string' ? value : key
                            );
                            console.log(`Topic ${index + 1} converted from object entries:`, keywords);
                        } catch (e) {
                            console.error(`Failed to convert topic ${index + 1} from object:`, e);
                        }
                    }
                } else if (typeof topic === 'string') {
                    // Format 5: Single string (possibly comma separated)
                    keywords = topic.split(',').map(word => word.trim());
                    console.log(`Topic ${index + 1} is a string, split into:`, keywords);
                }
                
                if (keywords && keywords.length > 0) {
                    topicKeywords.textContent = keywords.join(', ');
                } else {
                    topicKeywords.textContent = 'No keywords available';
                    console.warn('Topic format not recognized:', topic);
                }
                
                topicElement.appendChild(topicHeader);
                topicElement.appendChild(topicKeywords);
                
                // Insert before the no-topics message
                listGroup.insertBefore(topicElement, noTopicsMessage);
            });
        } else {
            const noTopicsMessage = document.getElementById('no-topics-message');
            noTopicsMessage.classList.remove('d-none');
            console.log('No topics available in data:', data.topics);
        }
    }
    
    // Function to create sentiment summary
    function createSentimentSummary(data, topic, subreddit) {
        const summaryElement = document.getElementById('sentiment-summary');
        if (!summaryElement) return;
        
        // Determine dominant sentiment
        let dominantSentiment = 'neutral';
        let dominantPercentage = data.neutral.percentage;
        
        if (data.positive.percentage > dominantPercentage) {
            dominantSentiment = 'positive';
            dominantPercentage = data.positive.percentage;
        }
        
        if (data.negative.percentage > dominantPercentage) {
            dominantSentiment = 'negative';
            dominantPercentage = data.negative.percentage;
        }
        
        // Create sentiment text with proper icon
        let icon = '';
        let colorClass = '';
        
        if (dominantSentiment === 'positive') {
            icon = '<i class="fas fa-thumbs-up text-success me-1"></i>';
            colorClass = 'text-success';
        } else if (dominantSentiment === 'negative') {
            icon = '<i class="fas fa-thumbs-down text-danger me-1"></i>';
            colorClass = 'text-danger';
        } else {
            icon = '<i class="fas fa-balance-scale text-secondary me-1"></i>';
            colorClass = 'text-secondary';
        }
        
        // Create summary message
        const message = `
            ${icon} The overall sentiment towards "${topic}" in r/${subreddit} is 
            <span class="${colorClass} fw-bold">${dominantSentiment}</span> 
            (${dominantPercentage}%).
        `;
        
        summaryElement.innerHTML = message;
    }
    
    // Function to create statistical analysis summary
    function createAnalysisSummary(data, topic, subreddit) {
        const summaryElement = document.getElementById('analysis-summary');
        if (!summaryElement) return;
        
        // Clear previous content
        summaryElement.innerHTML = '';
        
        // Calculate sentiment ratio
        const posToNegRatio = data.negative.count === 0 ? 
            data.positive.count : 
            (data.positive.count / data.negative.count).toFixed(2);
        
        // Calculate year span if available
        let yearSpan = 'N/A';
        let yearRange = [];
        
        if (data.by_year && data.by_year.length > 0) {
            const years = data.by_year.map(item => item.year);
            const minYear = Math.min(...years);
            const maxYear = Math.max(...years);
            yearSpan = maxYear - minYear + 1;
            yearRange = [minYear, maxYear];
        }
        
        // Calculate total interactions (comments)
        const totalComments = data.total;
        
        // Determine sentiment volatility (standard deviation of sentiment percentages over years)
        let volatility = 'N/A';
        if (data.by_year && data.by_year.length > 1) {
            // Calculate positive sentiment percentages by year
            const posPercentages = data.by_year.map(item => {
                const total = item.positive + item.negative + item.neutral;
                return total > 0 ? (item.positive / total * 100) : 0;
            });
            
            // Calculate mean
            const mean = posPercentages.reduce((sum, val) => sum + val, 0) / posPercentages.length;
            
            // Calculate variance
            const variance = posPercentages.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / posPercentages.length;
            
            // Calculate standard deviation (volatility)
            volatility = Math.sqrt(variance).toFixed(2) + '%';
        }
        
        // Create stat cards
        const cards = [
            createStatCard('Total Comments', totalComments, 'comment', 'primary'),
            createStatCard('Positive to Negative Ratio', posToNegRatio, 'scale-balanced', 'success'),
            createStatCard('Time Period', yearRange.length ? `${yearRange[0]} - ${yearRange[1]}` : 'N/A', 'calendar', 'info'),
            createStatCard('Sentiment Volatility', volatility, 'chart-line', 'warning')
        ];
        
        // Add cards to summary element
        cards.forEach(card => summaryElement.appendChild(card));
        
        // Add a text analysis of the results
        const analysisText = document.createElement('div');
        analysisText.className = 'col-12 mt-3';
        
        let analysisContent = `<p class="mb-0">The sentiment analysis for "${topic}" in r/${subreddit} shows `;
        
        // Add information about positive/negative ratio
        if (posToNegRatio > 2) {
            analysisContent += `a strongly positive reaction with ${posToNegRatio}× more positive than negative comments. `;
        } else if (posToNegRatio > 1) {
            analysisContent += `a somewhat positive reaction with ${posToNegRatio}× more positive than negative comments. `;
        } else if (posToNegRatio < 0.5) {
            analysisContent += `a strongly negative reaction with ${(1/posToNegRatio).toFixed(2)}× more negative than positive comments. `;
        } else if (posToNegRatio < 1) {
            analysisContent += `a somewhat negative reaction with ${(1/posToNegRatio).toFixed(2)}× more negative than positive comments. `;
        } else {
            analysisContent += `a balanced distribution between positive and negative sentiments. `;
        }
        
        // Add information about time period if available
        if (yearRange.length) {
            analysisContent += `The data spans from ${yearRange[0]} to ${yearRange[1]}`;
            
            // Add information about volatility if available
            if (volatility !== 'N/A') {
                if (parseFloat(volatility) > 15) {
                    analysisContent += ` with high volatility (${volatility}), indicating significant shifts in sentiment over time.`;
                } else if (parseFloat(volatility) > 5) {
                    analysisContent += ` with moderate volatility (${volatility}), suggesting some changes in sentiment over time.`;
                } else {
                    analysisContent += ` with low volatility (${volatility}), suggesting stable sentiment over time.`;
                }
            }
        }
        
        analysisContent += `</p>`;
        analysisText.innerHTML = analysisContent;
        summaryElement.appendChild(analysisText);
    }
    
    // Function to create a stat card
    function createStatCard(title, value, icon, colorClass) {
        const card = document.createElement('div');
        card.className = 'col-md-3 col-sm-6 mb-3';
        
        card.innerHTML = `
            <div class="border rounded p-3 h-100 d-flex flex-column">
                <div class="d-flex align-items-center mb-2">
                    <i class="fas fa-${icon} text-${colorClass} me-2"></i>
                    <span class="text-muted small">${title}</span>
                </div>
                <div class="h4 mt-auto mb-0 fw-bold">${value}</div>
            </div>
        `;
        
        return card;
    }
    
    // Function to update sentiment table
    function updateSentimentTable(data) {
        // Clear existing table
        sentimentTable.innerHTML = '';
        
        // Create table rows
        const rows = [
            createTableRow('Positive', data.positive.count, data.positive.percentage, 'success'),
            createTableRow('Negative', data.negative.count, data.negative.percentage, 'danger'),
            createTableRow('Neutral', data.neutral.count, data.neutral.percentage, 'secondary'),
            createTableRow('Total', data.total, 100, 'primary')
        ];
        
        // Add rows to table
        rows.forEach(row => sentimentTable.appendChild(row));
    }
    
    // Function to create table row
    function createTableRow(label, count, percentage, colorClass) {
        const row = document.createElement('tr');
        
        const labelCell = document.createElement('td');
        const labelBadge = document.createElement('span');
        labelBadge.className = `badge bg-${colorClass} me-1`;
        labelBadge.textContent = label;
        labelCell.appendChild(labelBadge);
        
        const countCell = document.createElement('td');
        countCell.textContent = count;
        
        const percentageCell = document.createElement('td');
        percentageCell.textContent = `${percentage}%`;
        
        row.appendChild(labelCell);
        row.appendChild(countCell);
        row.appendChild(percentageCell);
        
        return row;
    }
    
    // Function to create doughnut chart
    function createDoughnutChart(data) {
        const ctx = document.getElementById('sentiment-doughnut').getContext('2d');
        
        // Destroy previous chart if it exists
        if (doughnutChart) {
            doughnutChart.destroy();
        }
        
        // Create new chart
        doughnutChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Negative', 'Neutral'],
                datasets: [{
                    data: [
                        data.positive.count,
                        data.negative.count,
                        data.neutral.count
                    ],
                    backgroundColor: [
                        '#198754', // Bootstrap success
                        '#dc3545', // Bootstrap danger
                        '#6c757d'  // Bootstrap secondary
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const percentage = context.raw / data.total * 100;
                                return `${label}: ${value} (${percentage.toFixed(1)}%)`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Function to create emotions chart
    function createEmotionsChart(emotionsData) {
        const canvas = document.getElementById('emotions-chart');
        
        // Destroy existing chart if it exists
        if (emotionsChart) {
            emotionsChart.destroy();
        }
        
        const ctx = canvas.getContext('2d');
        
        // Convert emotions data to arrays for the chart
        const emotions = Object.keys(emotionsData);
        const intensities = Object.values(emotionsData);
        
        // Calculate colors for each emotion (gradient from light to saturated)
        const colors = emotions.map(emotion => {
            // Map common emotions to appropriate colors
            switch(emotion.toLowerCase()) {
                case 'joy':
                case 'happiness':
                case 'excited':
                    return 'rgba(255, 193, 7, 0.8)'; // warning/yellow
                case 'sadness':
                case 'sorrow':
                case 'grief':
                    return 'rgba(13, 110, 253, 0.8)'; // primary/blue
                case 'anger':
                case 'rage':
                case 'fury':
                    return 'rgba(220, 53, 69, 0.8)'; // danger/red
                case 'fear':
                case 'anxiety':
                case 'worry':
                    return 'rgba(108, 117, 125, 0.8)'; // secondary/gray
                case 'surprise':
                case 'amazement':
                case 'astonishment':
                    return 'rgba(111, 66, 193, 0.8)'; // purple
                case 'disgust':
                case 'dislike':
                case 'distaste':
                    return 'rgba(40, 167, 69, 0.8)'; // success/green
                case 'trust':
                case 'acceptance':
                case 'admiration':
                    return 'rgba(23, 162, 184, 0.8)'; // info/cyan
                default:
                    return 'rgba(0, 123, 255, 0.8)'; // default blue
            }
        });
        
        // Create new chart
        emotionsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: emotions,
                datasets: [{
                    label: 'Emotion Intensity',
                    data: intensities,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2.5,
                plugins: {
                    title: {
                        display: true,
                        text: 'Detected Emotions',
                        font: {
                            size: 14
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    // Convert to percentage
                                    label += Math.round(context.parsed.y * 100) + '%';
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Intensity'
                        }
                    }
                }
            }
        });
    }
    
    // Function to create timeline chart
    function createTimelineChart(timeData, timeUnit = 'year') {
        // Ensure timeData is an array (defensive coding)
        if (!Array.isArray(timeData)) {
            console.error('Expected timeData to be an array but got:', timeData);
            // Try to convert object to array if possible
            if (timeData && typeof timeData === 'object') {
                const newTimeData = [];
                // For year data
                if (timeUnit === 'year') {
                    Object.entries(timeData).forEach(([year, data]) => {
                        if (typeof data === 'object') {
                            newTimeData.push({
                                year: parseInt(year),
                                positive: data.positive || 0,
                                negative: data.negative || 0,
                                neutral: data.neutral || 0
                            });
                        }
                    });
                } 
                // For month data
                else {
                    Object.entries(timeData).forEach(([month, data]) => {
                        if (typeof data === 'object') {
                            newTimeData.push({
                                month: month,
                                positive: data.positive || 0,
                                negative: data.negative || 0,
                                neutral: data.neutral || 0
                            });
                        }
                    });
                }
                if (newTimeData.length > 0) {
                    console.log('Converted timeData object to array:', newTimeData);
                    timeData = newTimeData;
                } else {
                    console.error('Failed to convert timeData. Using empty array.');
                    timeData = [];
                }
            } else {
                console.error('Invalid timeData format, using empty array');
                timeData = [];
            }
        }
        
        const canvas = document.getElementById('timeline-chart');
        
        // Destroy existing chart if it exists
        if (timelineChart) {
            timelineChart.destroy();
        }
        
        const ctx = canvas.getContext('2d');
        const chartControls = document.getElementById('chart-controls');
        
        // Format data for chart
        let labels = [], positiveData = [], negativeData = [], neutralData = [];
        
        try {
            if (timeData.length === 0) {
                console.log('No time data available, using empty arrays');
            }
            else if (timeUnit === 'month') {
                // Format month data
                labels = timeData.map(item => {
                    if (!item || !item.month) {
                        console.warn('Invalid month item:', item);
                        return 'Unknown';
                    }
                    const [year, month] = item.month.split('-');
                    return `${year}-${month}`;
                });
                positiveData = timeData.map(item => item && typeof item.positive === 'number' ? item.positive : 0);
                negativeData = timeData.map(item => item && typeof item.negative === 'number' ? item.negative : 0);
                neutralData = timeData.map(item => item && typeof item.neutral === 'number' ? item.neutral : 0);
            } else {
                // Format year data
                labels = timeData.map(item => {
                    if (!item || (item.year === undefined)) {
                        console.warn('Invalid year item:', item);
                        return 'Unknown';
                    }
                    return item.year;
                });
                positiveData = timeData.map(item => item && typeof item.positive === 'number' ? item.positive : 0);
                negativeData = timeData.map(item => item && typeof item.negative === 'number' ? item.negative : 0);
                neutralData = timeData.map(item => item && typeof item.neutral === 'number' ? item.neutral : 0);
            }
        } catch (error) {
            console.error('Error formatting timeline data:', error);
        }
        
        // Calculate totals for percentage view - with defensive coding to handle potential bad data
        const dataTotals = timeData.map(item => {
            if (!item) return 0;
            const pos = typeof item.positive === 'number' ? item.positive : 0;
            const neg = typeof item.negative === 'number' ? item.negative : 0;
            const neu = typeof item.neutral === 'number' ? item.neutral : 0;
            return pos + neg + neu;
        });
        
        const positivePercentage = timeData.map((item, index) => {
            if (!item || dataTotals[index] === 0) return 0;
            const pos = typeof item.positive === 'number' ? item.positive : 0;
            return parseFloat((pos / dataTotals[index] * 100).toFixed(1));
        });
        
        const negativePercentage = timeData.map((item, index) => {
            if (!item || dataTotals[index] === 0) return 0;
            const neg = typeof item.negative === 'number' ? item.negative : 0;
            return parseFloat((neg / dataTotals[index] * 100).toFixed(1));
        });
        
        const neutralPercentage = timeData.map((item, index) => {
            if (!item || dataTotals[index] === 0) return 0;
            const neu = typeof item.neutral === 'number' ? item.neutral : 0;
            return parseFloat((neu / dataTotals[index] * 100).toFixed(1));
        });
        
        // Find trend by comparing earliest and latest data points (if more than one exists)
        let trendAnalysis = "";
        if (timeData.length > 1) {
            try {
                const firstPoint = timeData[0] || { positive: 0, negative: 0, neutral: 0 };
                const lastPoint = timeData[timeData.length - 1] || { positive: 0, negative: 0, neutral: 0 };
                
                // Safely get values
                const firstPos = typeof firstPoint.positive === 'number' ? firstPoint.positive : 0;
                const firstNeg = typeof firstPoint.negative === 'number' ? firstPoint.negative : 0;
                const firstNeu = typeof firstPoint.neutral === 'number' ? firstPoint.neutral : 0;
                const firstTotal = firstPos + firstNeg + firstNeu || 1; // Avoid division by zero
                
                const lastPos = typeof lastPoint.positive === 'number' ? lastPoint.positive : 0;
                const lastNeg = typeof lastPoint.negative === 'number' ? lastPoint.negative : 0;
                const lastNeu = typeof lastPoint.neutral === 'number' ? lastPoint.neutral : 0;
                const lastTotal = lastPos + lastNeg + lastNeu || 1; // Avoid division by zero
                
                const firstPositivePerc = (firstPos / firstTotal) * 100;
                const lastPositivePerc = (lastPos / lastTotal) * 100;
                
                const firstNegativePerc = (firstNeg / firstTotal) * 100;
                const lastNegativePerc = (lastNeg / lastTotal) * 100;
                
                const positiveTrend = lastPositivePerc - firstPositivePerc;
                const negativeTrend = lastNegativePerc - firstNegativePerc;
                
                // Create a trend indicator
                const trendElement = document.createElement('div');
                trendElement.className = 'trend-analysis mt-2 small';
            
                if (Math.abs(positiveTrend) > 5 || Math.abs(negativeTrend) > 5) {
                    // Get first/last time label for trend analysis text
                    const firstLabel = labels[0];
                    const lastLabel = labels[labels.length - 1];
                    
                    if (positiveTrend > 5) {
                        trendElement.innerHTML = `<i class="fas fa-arrow-trend-up text-success me-1"></i> Positive sentiment increased by ${positiveTrend.toFixed(1)}% from ${firstLabel} to ${lastLabel}`;
                    } else if (positiveTrend < -5) {
                        trendElement.innerHTML = `<i class="fas fa-arrow-trend-down text-danger me-1"></i> Positive sentiment decreased by ${Math.abs(positiveTrend).toFixed(1)}% from ${firstLabel} to ${lastLabel}`;
                    } else if (negativeTrend > 5) {
                        trendElement.innerHTML = `<i class="fas fa-arrow-trend-up text-danger me-1"></i> Negative sentiment increased by ${negativeTrend.toFixed(1)}% from ${firstLabel} to ${lastLabel}`;
                    } else if (negativeTrend < -5) {
                        trendElement.innerHTML = `<i class="fas fa-arrow-trend-down text-success me-1"></i> Negative sentiment decreased by ${Math.abs(negativeTrend).toFixed(1)}% from ${firstLabel} to ${lastLabel}`;
                    }
                    
                    // Add trend element to the chart container
                    const chartContainer = document.getElementById('chart-container');
                    
                    // Remove any existing trend analysis
                    const existingTrend = chartContainer.querySelector('.trend-analysis');
                    if (existingTrend) {
                        existingTrend.remove();
                    }
                    
                    chartContainer.appendChild(trendElement);
                }
            } catch (error) {
                console.error('Error calculating sentiment trend:', error);
            }
        }
        
        // Clear chart controls
        chartControls.innerHTML = '';
        
        // Create new chart
        timelineChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Positive',
                        data: positiveData,
                        backgroundColor: '#198754',
                        stack: 'Stack 0'
                    },
                    {
                        label: 'Negative',
                        data: negativeData,
                        backgroundColor: '#dc3545',
                        stack: 'Stack 0'
                    },
                    {
                        label: 'Neutral',
                        data: neutralData,
                        backgroundColor: '#6c757d',
                        stack: 'Stack 0'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,  // Adjust this to control height
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: timeUnit === 'month' ? 'Month' : 'Year'
                        }
                    },
                    y: {
                        stacked: true,
                        title: {
                            display: true,
                            text: 'Number of Comments'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Sentiment Trend Over Time',
                        font: {
                            size: 14
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            padding: 10
                        }
                    },
                    tooltip: {
                        callbacks: {
                            footer: function(tooltipItems) {
                                const total = tooltipItems.reduce((sum, item) => sum + item.parsed.y, 0);
                                return `Total: ${total}`;
                            }
                        }
                    }
                }
            }
        });
        
        // Add chart type toggle button
        const toggleCountPercentBtn = document.createElement('button');
        toggleCountPercentBtn.className = 'btn btn-sm btn-outline-secondary';
        toggleCountPercentBtn.textContent = 'Show as Percentage';
        toggleCountPercentBtn.addEventListener('click', function() {
            const isShowingCounts = toggleCountPercentBtn.textContent.includes('Percentage');
            
            if (isShowingCounts) {
                // Switch to percentage view
                timelineChart.data.datasets[0].data = positivePercentage;
                timelineChart.data.datasets[1].data = negativePercentage;
                timelineChart.data.datasets[2].data = neutralPercentage;
                timelineChart.options.scales.y.title.text = 'Percentage of Comments';
                timelineChart.options.plugins.title.text = 'Sentiment Distribution (%) by Year';
                toggleCountPercentBtn.textContent = 'Show as Count';
            } else {
                // Switch back to count view
                timelineChart.data.datasets[0].data = positiveData;
                timelineChart.data.datasets[1].data = negativeData;
                timelineChart.data.datasets[2].data = neutralData;
                timelineChart.options.scales.y.title.text = 'Number of Comments';
                timelineChart.options.plugins.title.text = 'Sentiment Trend Over Time';
                toggleCountPercentBtn.textContent = 'Show as Percentage';
            }
            
            timelineChart.update();
        });
        
        // Add chart type toggle button (bar/line)
        const toggleChartTypeBtn = document.createElement('button');
        toggleChartTypeBtn.className = 'btn btn-sm btn-outline-secondary ms-2';
        toggleChartTypeBtn.textContent = 'Show as Line';
        toggleChartTypeBtn.addEventListener('click', function() {
            const isBar = timelineChart.config.type === 'bar';
            
            if (isBar) {
                timelineChart.config.type = 'line';
                timelineChart.options.scales.y.stacked = false;
                timelineChart.data.datasets.forEach(dataset => {
                    dataset.fill = false;
                    dataset.tension = 0.1;
                });
                toggleChartTypeBtn.textContent = 'Show as Bar';
            } else {
                timelineChart.config.type = 'bar';
                timelineChart.options.scales.y.stacked = true;
                timelineChart.data.datasets.forEach(dataset => {
                    delete dataset.fill;
                    delete dataset.tension;
                });
                toggleChartTypeBtn.textContent = 'Show as Line';
            }
            
            timelineChart.update();
        });
        
        // Add buttons to chart controls
        chartControls.appendChild(toggleCountPercentBtn);
        chartControls.appendChild(toggleChartTypeBtn);
    }
    
    // Function to display comment samples
    function displayComments(sentimentType, samples, container) {
        // Clear container
        container.innerHTML = '';
        
        // Check if there are samples
        if (!samples || samples.length === 0) {
            const emptyMessage = document.createElement('p');
            emptyMessage.className = 'text-muted text-center';
            emptyMessage.textContent = `No ${sentimentType} comments found`;
            container.appendChild(emptyMessage);
            return;
        }
        
        // Create and append comment cards
        samples.forEach(comment => {
            const commentCard = createCommentCard(comment, sentimentType);
            container.appendChild(commentCard);
        });
    }
    
    // Function to create comment card
    function createCommentCard(comment, sentimentType) {
        const card = document.createElement('div');
        card.className = `card comment-card ${sentimentType} mb-2`;
        
        const cardBody = document.createElement('div');
        cardBody.className = 'card-body';
        
        const commentText = document.createElement('p');
        commentText.className = 'mb-2';
        commentText.textContent = comment.text;
        
        const commentFooter = document.createElement('div');
        commentFooter.className = 'd-flex justify-content-between align-items-center';
        
        const commentDate = document.createElement('span');
        commentDate.className = 'comment-date';
        commentDate.innerHTML = `<i class="far fa-calendar-alt me-1"></i> ${comment.date}`;
        
        const commentScore = document.createElement('span');
        commentScore.className = 'comment-score';
        commentScore.innerHTML = `<i class="fas fa-arrow-up me-1"></i> ${comment.score}`;
        
        commentFooter.appendChild(commentDate);
        commentFooter.appendChild(commentScore);
        
        cardBody.appendChild(commentText);
        cardBody.appendChild(commentFooter);
        card.appendChild(cardBody);
        
        return card;
    }
    
    // Function to create word frequency display
    function createWordFrequencyDisplay(sentimentType, wordFreqData) {
        const container = document.getElementById(`${sentimentType}-keywords`);
        if (!container) return;
        
        // Clear container
        container.innerHTML = '';
        
        // Check if there is word frequency data
        if (!wordFreqData || Object.keys(wordFreqData).length === 0) {
            const emptyMessage = document.createElement('p');
            emptyMessage.className = 'text-muted text-center';
            emptyMessage.textContent = `No common words found in ${sentimentType} comments`;
            container.appendChild(emptyMessage);
            return;
        }
        
        // Convert word frequency object to array and sort by frequency
        const wordFreqArray = Object.entries(wordFreqData)
            .map(([word, count]) => ({ word, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 25); // Take top 25 words
        
        // Find max frequency for scaling
        const maxFreq = Math.max(...wordFreqArray.map(item => item.count));
        
        // Create word cloud
        const wordCloud = document.createElement('div');
        wordCloud.className = 'word-cloud-container d-flex flex-wrap justify-content-center';
        
        // Add words with sizes based on frequency
        wordFreqArray.forEach(item => {
            const wordElement = document.createElement('span');
            
            // Calculate size based on frequency (between 0.8em and 2.0em)
            const fontSize = 0.8 + (item.count / maxFreq) * 1.2;
            
            // Calculate opacity based on frequency (between 0.7 and 1.0)
            const opacity = 0.7 + (item.count / maxFreq) * 0.3;
            
            // Set color based on sentiment type
            let color;
            if (sentimentType === 'positive') {
                color = `rgba(25, 135, 84, ${opacity})`;  // Bootstrap success with opacity
            } else if (sentimentType === 'negative') {
                color = `rgba(220, 53, 69, ${opacity})`;  // Bootstrap danger with opacity
            } else {
                color = `rgba(108, 117, 125, ${opacity})`;  // Bootstrap secondary with opacity
            }
            
            // Apply styles
            wordElement.style.fontSize = `${fontSize}em`;
            wordElement.style.color = color;
            wordElement.style.margin = '0.3em';
            wordElement.style.padding = '0.2em';
            wordElement.style.display = 'inline-block';
            wordElement.style.cursor = 'default';
            
            // Add tooltip with count
            wordElement.setAttribute('title', `${item.word}: ${item.count} occurrences`);
            wordElement.setAttribute('data-bs-toggle', 'tooltip');
            wordElement.setAttribute('data-bs-placement', 'top');
            
            // Add word text
            wordElement.textContent = item.word;
            
            // Add to cloud
            wordCloud.appendChild(wordElement);
        });
        
        // Add word cloud to container
        container.appendChild(wordCloud);
        
        // Initialize tooltips
        const tooltips = [].slice.call(container.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltips.map(tooltip => new bootstrap.Tooltip(tooltip));
    }
    
    // Function to load available datasets for the dropdown
    
    // Function to create emotions chart
    function createEmotionsChart(emotionsData) {
        const canvas = document.getElementById('emotions-chart');
        
        // Destroy existing chart if it exists
        if (emotionsChart) {
            emotionsChart.destroy();
        }
        
        const ctx = canvas.getContext('2d');
        
        // Convert emotions data to arrays for the chart
        const emotions = Object.keys(emotionsData);
        const intensities = Object.values(emotionsData);
        
        // Calculate colors for each emotion (gradient from light to saturated)
        const colors = emotions.map(emotion => {
            // Map common emotions to appropriate colors
            switch(emotion.toLowerCase()) {
                case 'joy':
                case 'happiness':
                case 'excited':
                    return 'rgba(255, 193, 7, 0.8)'; // warning/yellow
                case 'sadness':
                case 'sorrow':
                case 'grief':
                    return 'rgba(13, 110, 253, 0.8)'; // primary/blue
                case 'anger':
                case 'rage':
                case 'fury':
                    return 'rgba(220, 53, 69, 0.8)'; // danger/red
                case 'fear':
                case 'anxiety':
                case 'worry':
                    return 'rgba(108, 117, 125, 0.8)'; // secondary/gray
                case 'surprise':
                case 'amazement':
                case 'astonishment':
                    return 'rgba(111, 66, 193, 0.8)'; // purple
                case 'disgust':
                case 'dislike':
                case 'distaste':
                    return 'rgba(40, 167, 69, 0.8)'; // success/green
                case 'trust':
                case 'acceptance':
                case 'admiration':
                    return 'rgba(23, 162, 184, 0.8)'; // info/cyan
                default:
                    return 'rgba(0, 123, 255, 0.8)'; // default blue
            }
        });
        
        // Create new chart
        emotionsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: emotions,
                datasets: [{
                    label: 'Emotion Intensity',
                    data: intensities,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2.5,
                plugins: {
                    title: {
                        display: true,
                        text: 'Detected Emotions',
                        font: {
                            size: 14
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    // Convert to percentage
                                    label += Math.round(context.parsed.y * 100) + '%';
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        },
                        title: {
                            display: true,
                            text: 'Intensity'
                        }
                    }
                }
            }
        });
    }
    
    function loadAvailableDatasets() {
        fetch('http://127.0.0.1:5000/get_available_datasets')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to load available datasets');
                }
                return response.json();
            })
            .then(data => {
                if (data.datasets && data.datasets.length > 0) {
                    // Clear existing options (except the default one)
                    const options = Array.from(datasetSelect.options);
                    for (let i = options.length - 1; i >= 0; i--) {
                        if (options[i].value !== '') {
                            datasetSelect.remove(i);
                        }
                    }
                    
                    // Add new options
                    data.datasets.forEach(dataset => {
                        const option = document.createElement('option');
                        option.value = dataset.id;
                        option.textContent = dataset.display_name;
                        option.dataset.topic = dataset.topic;
                        option.dataset.subreddit = dataset.subreddit;
                        datasetSelect.appendChild(option);
                    });
                    
                    // Update subreddit options from first dataset
                    if (datasetSelect.options.length > 1) {
                        updateSubredditOptions();
                    }
                } else {
                    // No datasets available, add a disabled option
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'No datasets available';
                    option.disabled = true;
                    datasetSelect.innerHTML = '';
                    datasetSelect.appendChild(option);
                }
            })
            .catch(error => {
                console.error('Error loading datasets:', error);
                // Add a disabled option indicating error
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'Error loading datasets';
                option.disabled = true;
                datasetSelect.innerHTML = '';
                datasetSelect.appendChild(option);
            });
    }
    
    // Function to update subreddit options based on the selected dataset
    function updateSubredditOptions() {
        const datasetId = datasetSelect.value;
        
        if (!datasetId) return;
        
        // Show loading state
        setDatasetLoadingState(true);
        
        // Make API request to get dataset info with all subreddits
        fetch('http://127.0.0.1:5000/analyze_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ dataset_id: datasetId })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'An error occurred during dataset analysis');
                });
            }
            return response.json();
        })
        .then(data => {
            // Extract unique subreddits from the data
            if (data.subreddits && data.subreddits.length > 0) {
                // Clear existing options (except "All Subreddits")
                const options = Array.from(filterSubreddit.options);
                for (let i = options.length - 1; i >= 0; i--) {
                    if (options[i].value !== 'all') {
                        filterSubreddit.remove(i);
                    }
                }
                
                // Store subreddit data for filtering
                availableSubreddits = data.subreddits;
                
                // Add new options sorted by total comment count
                data.subreddits
                    .sort((a, b) => b.total - a.total)
                    .forEach(subreddit => {
                        const option = document.createElement('option');
                        option.value = subreddit.name;
                        option.textContent = `r/${subreddit.name} (${subreddit.total} comments)`;
                        filterSubreddit.appendChild(option);
                    });
            }
        })
        .catch(error => {
            console.error('Error loading subreddits:', error);
        })
        .finally(() => {
            setDatasetLoadingState(false);
        });
    }
    
    // Function to analyze a dataset
    function analyzeDataset(e) {
        e.preventDefault();
        
        // Get selected dataset
        const datasetId = datasetSelect.value;
        
        // Validate input
        if (!datasetId) {
            showError('Please select a dataset to analyze');
            return;
        }
        
        // Get filter values
        const topicFilter = filterTopic.value.trim().toLowerCase();
        const subredditFilter = filterSubreddit.value;
        const includeEmotion = document.getElementById('dataset-include-emotion').checked;
        
        // Show loading state
        setDatasetLoadingState(true);
        hideError();
        
        // Get dataset details from the selected option for display
        const selectedOption = datasetSelect.options[datasetSelect.selectedIndex];
        const datasetDisplay = selectedOption.textContent;
        
        // Extract topic and subreddit from display name (format: "Topic (r/subreddit)")
        let topic = datasetDisplay;
        let subreddit = 'all';
        
        const match = datasetDisplay.match(/^(.+) \((.+)\)$/);
        if (match && match.length === 3) {
            topic = match[1].trim();
            subreddit = match[2].replace('r/', '').trim();
            if (subreddit === 'All Subreddits') {
                subreddit = 'all';
            }
        }
        
        // Make API request
        fetch('http://127.0.0.1:5000/analyze_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: datasetId,
                topic_filter: topicFilter,
                subreddit_filter: subredditFilter,
                include_emotion: includeEmotion
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'An error occurred during dataset analysis');
                });
            }
            return response.json();
        })
        .then(data => {
            // Update display topic/subreddit based on filters if applied
            let displayTopic = topic;
            let displaySubreddit = subreddit;
            
            if (topicFilter) {
                displayTopic = `${topic} (filtered: "${topicFilter}")`;
            }
            
            if (subredditFilter !== 'all') {
                displaySubreddit = subredditFilter;
            }
            
            // Process and display results
            displayResults(data, displayTopic, displaySubreddit);
        })
        .catch(error => {
            console.error('Error:', error);
            showError(error.message || 'Failed to analyze dataset');
        })
        .finally(() => {
            setDatasetLoadingState(false);
        });
    }
    
    // Function to set dataset analysis loading state
    function setDatasetLoadingState(isLoading) {
        if (isLoading) {
            datasetBtnText.textContent = 'Analyzing...';
            datasetLoadingSpinner.classList.remove('d-none');
            datasetAnalyzeBtn.disabled = true;
        } else {
            datasetBtnText.textContent = 'Analyze Dataset';
            datasetLoadingSpinner.classList.add('d-none');
            datasetAnalyzeBtn.disabled = false;
        }
    }
    
    // Make selectMode function globally accessible for HTML onclick attributes
    window.selectMode = selectMode;
    
    // Dark Mode Functionality
    const themeToggleBtn = document.getElementById('theme-toggle');
    
    // Check if user has a theme preference stored
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        document.documentElement.setAttribute('data-bs-theme', savedTheme);
        updateThemeToggleButton(savedTheme);
    }
    
    // Handle theme toggle click
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-bs-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        
        // Update theme attribute
        document.documentElement.setAttribute('data-bs-theme', newTheme);
        
        // Save preference to localStorage
        localStorage.setItem('theme', newTheme);
        
        // Update button appearance
        updateThemeToggleButton(newTheme);
    });
    
    function updateThemeToggleButton(theme) {
        const icon = themeToggleBtn.querySelector('i');
        const text = themeToggleBtn.querySelector('span');
        
        if (theme === 'dark') {
            icon.className = 'fas fa-sun';
            text.textContent = 'Light Mode';
            themeToggleBtn.classList.remove('btn-outline-light');
            themeToggleBtn.classList.add('btn-outline-warning');
        } else {
            icon.className = 'fas fa-moon';
            text.textContent = 'Dark Mode';
            themeToggleBtn.classList.remove('btn-outline-warning');
            themeToggleBtn.classList.add('btn-outline-light');
        }
    }
});
