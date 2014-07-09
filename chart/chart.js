var margin = {top: 20, right: 20, bottom: 150, left: 40},
width = 960 - margin.left - margin.right,
height = 700 - margin.top - margin.bottom;

var x0 = d3.scale.ordinal()
    .rangeRoundBands([0, width], .1);

var x1 = d3.scale.ordinal();

var y = d3.scale.linear()
    .range([height, 0]);

var color = d3.scale.ordinal()
    .range(["#00CC33", "#000000", "#0000FF", "#6b486b", "#a05d56", "#d0743c", "#ff8c00"]);


var svg = null;

function createCheckboxes(data) {
    function createButton(value, callback) {
        var el = document.createElement('input');
        el.type = "button";
        el.value = value;
        el.addEventListener('click', callback);
        return el;
    }

    function createCheckbox(id, className, name, value, checked, callback) {
        var el = document.createElement('input');
        el.type = "checkbox";
        el.id = id;
        el.className = className;
        el.name = name;
        el.value = value;
        el.checked = checked ? 'checked' : '';
        el.addEventListener('change', callback);
        var lbl = document.createElement('label');
        lbl.appendChild(el);
        lbl.appendChild(document.createTextNode(id));
        return lbl;
    }


    // Add classifiers div
    var classifiers = document.createElement('div');
    classifiers.id = "classifiers";

    var classifiersHeader = document.createElement('h2');
    classifiersHeader.textContent = "Classifiers";
    classifiers.appendChild(classifiersHeader);

    classifiers.appendChild(
        createButton('Check All', function(el) {
            checkall();
            makeplot(false);
        })
    );
    classifiers.appendChild(
        createButton('Uncheck All', function(el) {
            uncheckall();
            makeplot(false);
        })
    );
    classifiers.appendChild(
        document.createElement('br')
    );

    for(var i = 0; i < data.length; i++){
        stuff = data[i];
        classifiers.appendChild(
            createCheckbox(stuff.Classifier, 'classifier_ctrl', 'button', 'val', true, function(el) {
                makeplot(false)
            })
        );

        if ((1+i) % 4 == 0) {
            classifiers.appendChild(document.createElement('br'));
        }
    };

    // Add metrics div
    var metrics = document.createElement('div');
    metrics.id = "metrics";

    var metricsHeader = document.createElement('h2');
    metricsHeader.textContent = "Metrics";
    metrics.appendChild(metricsHeader);

    metrics.appendChild(
        createCheckbox('Score', 'classifier_ctrl', 'value', 'stat', true, function(el) {
            makeplot(false);
        })
    );
    metrics.appendChild(
        createCheckbox('Train.Time', 'classifier_ctrl', 'value', 'stat', true, function(el) {
            makeplot(false);
        })
    );
    metrics.appendChild(
        createCheckbox('Test.Time', 'classifier_ctrl', 'value', 'stat', true, function(el) {
            makeplot(false);
        })
    );

    // add metrics and classifiers div to parent element
    var container = document.getElementById('chart');
    container.appendChild(classifiers);
    container.appendChild(metrics);
}

var xAxis = d3.svg.axis()
    .scale(x0)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left")
    .tickFormat(d3.format(".2s"));

var tip = d3.tip()
    .attr('class', 'd3-tip')
    .offset([-10, 0])
    .html(function(d) {
        return "<strong> Value:</strong> <span style='color:yellow'>" + Math.round(d.value*1000)/1000 + "</span>";
    });

makeplot = function(doCreateCheckboxes) {

    svg.selectAll("*").remove();
    d3.csv(g_dataFile, function(error, data) {
        if (doCreateCheckboxes) {
            createCheckboxes(data);
        }

        data = data.filter(function(row) {
            return document.getElementById(row['Classifier']).checked;
        });

        var valnames = d3.keys(data[0]).filter(function(key) {
            return key !== "Classifier" && document.getElementById(key).checked;
        });

        data.forEach(function(d) {
            d.names = valnames.map(function(name) {
                return {name: name, value: +d[name]};
            });
        });

        x0.domain(data.map(function(d) { return d.Classifier; }));
        x1.domain(valnames).rangeRoundBands([0, x0.rangeBand()]);
        y.domain([0, d3.max(data, function(d) {
            return d3.max(d.names, function(d) { return d.value; });
        })]);

        svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis)
            .selectAll("text")
            .attr("y", 0)
            .attr("x", 9)
            .attr("dy", ".35em")
            .attr("transform", "rotate(90)")
            .style("text-anchor", "start");

        svg.append("g")
            .attr("class", "y axis")
            .call(yAxis)
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 6)
            .attr("dy", ".71em")
            .style("text-anchor", "end")
            .text("value");

        var classifier = svg.selectAll(".Classifier")
            .data(data)
            .enter().append("g")
            .attr("class", "g")
            .attr("transform", function(d) { return "translate(" + x0(d.Classifier) + ",0)"; });

        classifier.selectAll("rect")
            .data(function(d) { return d.names; })
            .enter().append("rect")
            .attr("width", x1.rangeBand())
            .attr("x", function(d) { return x1(d.name); })
            .attr("y", function(d) { return y(d.value); })
            .attr("height", function(d) { return height - y(d.value); })
            .style("fill", function(d) { return color(d.name); })
            .on('mouseover',tip.show)
            .on('mouseout',tip.hide);

        var legend = svg.selectAll(".legend")
            .data(valnames.slice().reverse())
            .enter().append("g")
            .attr("class", "legend")
            .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

        legend.append("rect")
            .attr("x", width - 18)
            .attr("width", 18)
            .attr("height", 18)
            .style("fill", color);

        legend.append("text")
            .attr("x", width - 24)
            .attr("y", 9)
            .attr("dy", ".35em")
            .style("text-anchor", "end")
            .text(function(d) { return d; });
    });
};

$(document).ready(function() {

    svg = d3.select("body").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    svg.call(tip);

    makeplot(true);
});

checkall = function(){
    boxes = document.getElementsByName('button');
    for(var i = 0; i < boxes.length;i++){
        boxes[i].checked = true;
    }
};

uncheckall = function(){
    boxes = document.getElementsByName('button');
    for(var i = 0; i < boxes.length;i++){
        boxes[i].checked = false;
    }
};
