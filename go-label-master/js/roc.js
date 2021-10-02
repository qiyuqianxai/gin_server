var cards;
var current_deck_id;
var current_fpr = '';
var current_type;
var dataset;
var server_address;

function go_to_homepage() {
    window.location.href = server_address;
}

function isIn(el) {
    var bound = el.getBoundingClientRect();
    var clientHeight = window.innerHeight;
    return bound.top <= clientHeight;
}

function check() {
    cards = document.getElementById(current_deck_id).querySelectorAll('img');
    if (cards.length > 0) {
        Array.from(cards).slice(-1).forEach(function (el) {
            if (isIn(el)) {
                load_images();
            }
        })
    } else {
        load_images();
    }
}

function show_size() {
    var request = `/roc-size/${dataset}/${current_fpr}/${current_type}`;
    $.getJSON(request, function (result) {
        $("#size").text(result.size);
    });
}

function prefix(path) {
    if (path[0] === "/") {
        return "/static/path";
    } else {
        return "/performance-raw";
    }
}

function load_images() {
    window.onscroll = null;
    var loc = document.getElementById(current_deck_id).querySelectorAll('.row').length;
    var request = `/roc-data/${dataset}/${current_fpr}/${current_type}/${loc}`;
    $.getJSON(request, function (result) {
        var hd = result.hd;
        var xc = result.xc;
        var score = result.score;
        var label = result.label;
        var test_name = result.test_name;
        var top_hd = result.top_hd;
        var top_score = result.top_score;
        var hd_name = result.hd_name;
        var xc_name = result.xc_name;
        var ex_name = result.ex_name;
        // var threshold = result.threshold;
        for (var i = 0; i < hd.length; i++) {
            var top = "";
            if (current_type === "miss") {
                var top_pre = prefix(top_hd[i]);
                top = `<div class="col-lg-3 col-sm-3 card-col">
                <a href="${top_pre}/${top_hd[i]}" target="_blank" class="card">
                <img src="${top_pre}/${top_hd[i]}" class="card-img-top">
                <ul class="list-group list-group-flush">
                    <li class="list-group-item">${ex_name}</li>
                    <li class="list-group-item">score: ${top_score[i]}</li>
                </ul></a></div>`;
            }
            var xc_pre = prefix(xc[i]);
            var hd_pre = prefix(hd[i]);
            var s = `<div class="row" style="width: 100vw">
                <div class="col-lg-1 col-sm-1 card-col my-auto">
                    <ul class="list-group list-group-flush" style="margin-top: 5rem;">
                        <li class="list-group-item">${i + 1 + loc}</li>
                    </ul></div>
                <div class="col-lg-3 col-sm-3 card-col">
                    <a href="${xc_pre}/${xc[i]}" target="_blank" class="card">
                    <img src="${xc_pre}/${xc[i]}" class="card-img-top">
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item">${xc_name}</li>
                        <li class="list-group-item">test_name: ${test_name[i]}</li>
                        <li class="list-group-item">label: ${label[i]}</li>
                    </ul></a></div>
                <div class="col-lg-3 col-sm-3 card-col">
                    <a href="${hd_pre}/${hd[i]}" target="_blank" class="card">
                    <img src="${hd_pre}/${hd[i]}" class="card-img-top">
                    <ul class="list-group list-group-flush">
                        <li class="list-group-item">${hd_name}</li>
                        <li class="list-group-item">score: ${score[i]}</li>
                    </ul></a></div>
                ${top}</div>`;
            $(`#${current_deck_id}`).append(s);
        }
        if (window.onscroll == null) {
            window.onscroll = function () {
                check();
            }
        }
    })
}

function loadCard(im) {
    if (!im.src) {
        var source = im.dataset.src;
        im.src = source;
    }
}

// window.onload = 
window.onscroll = function () {
    check();
}

function show_only(x) {
    current_deck_id = `${current_fpr}-${x}-deck`;
    current_type = x;
    var y;
    if (x === 'high-neg') {
        y = 'low-pos';
    } else {
        y = 'high-neg';
    }
    $(`#${x}-btn`).each(function () {
        $(this).removeClass("btn-secondary");
        $(this).addClass("btn-primary");
    });
    $(`#${y}-btn`).each(function () {
        $(this).removeClass("btn-primary");
        $(this).addClass("btn-secondary");
    });
    $(`#${current_fpr}-${x}-deck`).each(function () {
        $(this).removeClass("sr-only");
    });
    $(`#${current_fpr}-${y}-deck`).each(function () {
        $(this).addClass("sr-only");
    });
    show_size();
    check();
}

function show_filter(x) {
    current_deck_id = `${x}-deck`;
    current_type = x;
    var y;
    if (x === 'filtered-pos') {
        y = 'filtered-neg';
    } else {
        y = 'filtered-pos';
    }
    $(`#${x}-btn`).each(function () {
        $(this).removeClass("btn-secondary");
        $(this).addClass("btn-primary");
    });
    $(`#${y}-btn`).each(function () {
        $(this).removeClass("btn-primary");
        $(this).addClass("btn-secondary");
    });
    $(`#${x}-deck`).each(function () {
        $(this).removeClass("sr-only");
    });
    $(`#${y}-deck`).each(function () {
        $(this).addClass("sr-only");
    });
    show_size();
    check();
}

function hide_all() {
    if (current_fpr !== '') {
        if (current_fpr === 'miss') {
            $("#miss-pair-deck").addClass("sr-only");
        } else if (current_fpr === 'filter') {
            $("#filtered-pair").addClass("sr-only");
            $("#xc-hd-btn").addClass("sr-only");
        } else {
            $("#pos-neg-btn").addClass("sr-only");
            $(`#${current_fpr}-high-neg-deck`).addClass("sr-only");
            $(`#${current_fpr}-low-pos-deck`).addClass("sr-only");
        }
    }
}

function show_pairs(fpr) {
    fpr = fpr.split('=')[1];
    current_fpr = fpr;
    if (fpr === 'miss') {
        current_type = "miss";
        $("#pos-neg-btn").addClass("sr-only");
        $(`#xc-hd-btn`).addClass("sr-only");
        $(`#miss-pair-deck`).removeClass("sr-only");
        $(`#fpr-page`).addClass("sr-only");
        $(`#filtered-pair`).addClass("sr-only");
        current_deck_id = "miss-pair-deck";
        $("#threshold").text("0");
        $("#fnr").text("0");
        show_size();
        check();
    } else if (fpr === 'filtered') {
        current_type = "filtered-pos";
        $("#pos-neg-btn").addClass("sr-only");
        $(`#xc-hd-btn`).removeClass("sr-only");
        $(`#filtered-pair`).removeClass("sr-only");
        $(`#miss-pair-deck`).addClass("sr-only");
        $(`#fpr-page`).addClass("sr-only");
        current_deck_id = "filtered-pos-deck";
        $("#threshold").text("0");
        $("#fnr").text("0");
        show_filter('filtered-neg');
    } else {
        current_type = "high-neg";
        $(`#xc-hd-btn`).addClass("sr-only");
        $("#pos-neg-btn").removeClass("sr-only");
        $(`#miss-pair-deck`).addClass("sr-only");
        $(`#filtered-pair`).addClass("sr-only");
        $(`#fpr-page`).removeClass("sr-only");
        current_deck_id = `${current_fpr}-high-neg-deck`;
        var thd = $(`#threshold-${current_fpr}`).text();
        $("#threshold").text(thd);
        var fnr = $(`#fnr-${current_fpr}`).text();
        $("#fnr").text(fnr);
        show_only('high-neg');
    }
}

$(document).ready(function () {
    $.ajaxSetup({ cache: false });
    server_address = window.location.origin;
    dataset = $("#dataset").html();
    $("#fprSelect").change(function () {
        $("#fprSelect option:selected").each(function () {
            hide_all();
            if ($(this).val() === "missed_top1_pairs") {
                show_pairs("fpr=miss");
            } else if ($(this).val().startsWith("fpr")) {
                show_pairs($(this).val());
            } else if ($(this).val() === "filtered") {
                show_pairs("fpr=filtered");
            }
        });
    }).trigger("change");
});

$(document).keyup(function (event) {
    switch (event.keyCode) {
        case 37: //left
            $('#previous').trigger("click");
            return;
        case 39: //right
            $('#next').trigger("click");
            return;
    }
});
