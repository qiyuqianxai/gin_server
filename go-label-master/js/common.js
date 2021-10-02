var cus_green = "rgb(33, 183, 67)";
var cus_red = "rgb(251, 3, 27)";
var current_id;
var job_start;
var job_end;
var total_job;
var server_address = "";
var name = "";
var dataset = "";
var wrongid = {};
var imode;
var img_type;
var is_body;
var lbs;
var current_rst;

function valid_session(func) {
    $.get( "/valid_session/" + name + "/" + dataset, function (data) {
        if (data === "valid") {
            func();
        } else {
            alert("session已过期.")
        }
    });
}

function gen_image_src(dataset, image_name, img_type="starbox", proxy=true) {
    if (image_name.includes("?")) {
        var q = image_name.indexOf("?");
        var s = image_name.indexOf("//") + 2;
        var server = image_name.substring(s, q);
        return `/image-query?server=${server}&${image_name.substring(q+1)}`;
    } else if (image_name[0] === "/") {
        return "/static/path" + image_name;
    } else if (image_name.includes("/")) {
        return "/static/imgs/" + image_name;
    } else if (image_name.includes(",")) { //performance
        if (proxy) {
            return "/performance/file/download?storageId=" + image_name
             + "&type=1&filename=tmp.jpg";
        } else {
            return "/performance-raw/" + image_name;
        }
    } else {
        if (proxy) {
            return "/starbox/starbox-prd-ai/" + image_name;
        } else {
            return "/starbox-raw/" + image_name;
        }
    }
}

function gen_body_image_src(dataset, image_name, meta, rect="", proxy=true) {
    if (typeof image_name === 'object') {
        var query = $.param(image_name);
        return "/starbox-body-raw?" + query;
    }
    if (image_name[0] === "/") {
        return "/static/path" + image_name;
    } else if (image_name.includes("/")) {
        return "/static/imgs/" + image_name;
    } else if (image_name.includes(",")) { //performance
        if (proxy) {
            return "/performance/file/download?storageId=" + image_name
             + "&type=1&filename=tmp.jpg";
        } else {
            return "/performance-raw/" + image_name;
        }
    } else {
        if (proxy) {
            return "/starbox/starbox-prd-ai/" + image_name;
        } else {
            return "/starbox/starbox-prd-ai/error"
        }
    }
}

function go_to_homepage() {
    window.location.href = server_address;
}

function gen_info(all=false) {
    var range;
    if (all) {
        range = "all";
    } else {
        range = "part";
    }
    return '/templates/' + range + '/' + dataset + '/' + current_id.toString();
}

function gen_query(all=true) {
    var range;
    if (all) {
        range = "all";
    } else {
        range = "part";
    }
    return '/templates-body/' + range + '/' + dataset + '/' + current_id.toString();
}

function get_current_rst(func) {
    $.getJSON("/get_current_rst/" + dataset + "/" + current_id.toString(),
    function (data) {
        current_rst = data.rst;
        func();
    });
}

function show_current_id() {
    $("#classid").text((current_id - job_start).toString() + ' / ' + total_job.toString());
    get_current_rst();
}

function refresh_page() {
    window.location.href = `/details/${dataset}/${is_body}/${current_id}`;
}

function update_current_id(func) {
    $.get( "/update_label/" + name + "/" + dataset +
        "/" + current_id.toString(),
        function (data) {
            if (data === "success") {
                func();
            } else {
                alert("更新 current_id 失败.")
            }
        });
}

function update_check_num() {
    $.getJSON("/get_num_label_checked/" + name + "/" + dataset,
    function( result ) {
        $("#check_num").text(result.num.toString());
    });
}

function goto(nid) {
    if (nid <= job_end && nid >= job_start) {
        current_id = nid;
        update_current_id(refresh_page);
    } else {
        //reset_actions();
        alert("超出任务范围.");
    }
}

function translate_rst(rst) {
    $.getJSON("/rst_to_label/" + dataset + "/"+ rst,
        function( data ) {
            var label = Number(data.label);
            if (label === -1) {
                alert("输入的 rst id 无效");
            } else {
                goto(label + job_start);
            }  
            $("#jump input").val("");
        });
}

function go_to_id() {
    var nid = $("#goid").val();
    if(isNaN(nid)){
        alert("请输入有效数字.");
    } else {
        translate_rst(nid);
    }
}
function go_to_label() {
    var nid = $("#goid").val();
    if(isNaN(nid)){
        alert("请输入有效数字.");
    } else {
        
        nid = Number(nid);
        if (nid < 0 || nid > total_job) {
            alert("输入的 label 无效.");
        } else {
            goto(nid + job_start);
        }  
        $("#jump input").val("");
        
    }
}

function enter_goid() {
    if (event.keyCode === 13) {
        go_to_id();
    }
}
function enter_go_label() {
    if (event.keyCode === 13) {
        go_to_label();
    }
}

function add_to_gallery(i, image_src, itype, origin_img_src, color=null, info=null) {
    var card_str, iclass, src;
    if (i < 30) {        
        src = "src";
    } else {
        // show = "sr-only";
        src = "data-src"
    }
    if (itype === 0) {
        iclass = "fface";
    } else if (itype === 1) {
        iclass = "fbody";
    } else {
        iclass = "other";
    }
    if (color === null) {
        card_str = "<div class=\"col-lg-2 col-sm-3 card-col " + iclass + 
            "\"><a href=\"" + origin_img_src + "\" target=\"_blank\" class=\"card\" id=\"face" + i.toString() +
            "\"><img " + src +"=\"" + image_src +
            "\" class=\"card-img-top\"></a></div>";
    } else if (info !== null) {
        var info_li = "";
        for (var j = 0; j < info.length; j++) {
            info_li += `<li class="list-group-item">${info[j]}</li>`;
        }
        card_str = "<div class=\"col-lg-2 col-sm-3 card-col " + iclass + 
            "\"><a href=\"" + origin_img_src + "\" target=\"_blank\" class=\"card\" id=\"face" + i.toString() +
            "\" style=\"border-width:8px;border-color:" + color + ";\"" +
            "\"><img " + src + "=\"" + image_src +
            "\" class=\"card-img-top\"><ul class=\"list-group list-group-flush\">" + 
            info_li + "</ul></a></div>";
    } else {
        card_str = "<div class=\"col-lg-2 col-sm-3 card-col " + iclass + 
            "\"><a href=\"" + origin_img_src + "\" target=\"_blank\" class=\"card\" id=\"face" + i.toString() +
            "\" style=\"border-width:8px;border-color:" + color + ";\"" +
            "><img " + src +"=\"" + image_src +
            "\" class=\"card-img-top\"></a></div>";
    }
    
    
    $("#gallery").append(card_str);
}

function add_to_neg_gallery(i, image_src, image_name, origin_img_src, red_border=false) {
    var card_str;
    if (!red_border) {
        card_str = "<div class=\"col-lg-2 col-sm-3 neg-card-col\" id=\"fface" + i.toString() +
            "\"><a href=\"#\" class=\"card\" id=" + i.toString() +
            "><b class=\"card-header\">" + i.toString() + "</b><img src=\"" + image_src +
            "\" class=\"card-img-top\"><div class=\"card-body\"><p class=\"card-title\">" +
            image_name + "</p></div></a><a href=\"" + origin_img_src +
            "\" class=\"btn btn-warning btn-sm cus-btn-block\" target=\"_blank\">查看原图</a></div>";
    } else {
        card_str = "<div class=\"col-lg-2 col-sm-3 neg-card-col\" id=\"fface" + i.toString() +
            "\"><a href=\"#\" class=\"card card-red\" id=" + i.toString() +
            "><b class=\"card-header\">" + i.toString() + "</b><img src=\"" + image_src +
            "\" class=\"card-img-top\"><div class=\"card-body\"><p class=\"card-title\">" +
            image_name + "</p></div></a><a href=\"" + origin_img_src +
            "\" class=\"btn btn-warning btn-sm cus-btn-block\" target=\"_blank\">查看原图</a></div>";
    }
    $("#neg-gallery").append(card_str);
}

function refresh_view(func) {
    $(".card-col").each(function () {
        $(this).remove();
    });
    $(".neg-card-col").each(function () {
        $(this).remove();
    });
    func();
}

function increase_current_id() {
    current_id = current_id + 1;
    localStorage["current"] = current_id;
    update_current_id(refresh_page);
}

function decrease_current_id() {
    current_id = current_id - 1;
    localStorage["current"] = current_id;
    update_current_id(refresh_page);
}

