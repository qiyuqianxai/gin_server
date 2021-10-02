var cards;

function shuffle(a) {
    var j, x, i;
    for (i = a.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = a[i];
        a[i] = a[j];
        a[j] = x;
    }
    return a;
}

function get_color_pool(ncolors) {
    var n = ncolors;
    // if (n < 12) {
    //     n = 12;
    // }
    var m = 2;
    while(Math.pow(m, 3) < (n+1)) {
        m += 1;
    }
    var pool = [];
    var sec = Math.ceil(255 / m);
    for (let i = 0; i < 255; i += sec) {
        for (let j = 0; j < 255; j += sec) {
            for (let k = 0; k < 255; k += sec) {
                let rgb = `rgba(${i}, ${j}, ${k}, .5)`;
                pool.push(rgb);
            }
        }
    }
    var random_pool = shuffle(pool);
    random_pool[0] = `rgba(255, 255, 255, .5)`;
    return random_pool;
}

function reset_actions() {
    $('#next').off('click', press_next);
    $('#previous').off('click', press_previous);
}

function center_view(request, action = "next") {
    $.getJSON(request, function( result ) {
        var imgpath = result.imgpath;
        var origin_img_path = result.org_sid;
        var color_ids = result.icolor;
        var rects = result.rects;
        var itype = result.itype;
        var camera_id = result.camera_id;
        var track_id = result.track_id;
        var timestamp = result.timestamp;
        var ts_beg = result.ts_beg;
        var ts_end = result.ts_end;
        var camera_name = result.camera_name;
        var score = result.score;
        var i;
        var image_name;
        var image_src;
        var origin_image_src;
        var meta, rect;
        var colors = get_color_pool(Number(result.ncolor));
        $("#classid").text((current_id - job_start).toString() + ' / ' + total_job.toString());
        $("#rst").text(current_rst + ": 共" + imgpath.length + "张");
        for (i = 0; i < imgpath.length; i++) {
            image_name = imgpath[i];
            org_image_name = origin_img_path[i];
            rect = rects[i];
            if (org_image_name === "null") {
                org_image_name = image_name;
                rect = "-1";
            } 
            meta = `${itype[i]},${camera_id[i]},${track_id[i]},${timestamp[i]},${ts_beg[i]},${ts_end[i]},${score[i]}`;
            image_src = gen_body_image_src(dataset, image_name, meta, rect, true);
            var params = {
                sid: image_name,
                meta: meta,
                rect: rect,
                org: org_image_name,
                camera: camera_name[i]
            }; 
            origin_image_src = gen_body_image_src(dataset, params, meta, rect, false);   
            add_to_gallery(i, image_src, Number(itype[i]), origin_image_src, colors[color_ids[i]]);
        }
        switch (imode) {
            case 'face':
                show_face_only();
                break;
            case 'body':
                show_body_only();
                break;
            case 'hybrid':
                show_hybrid(function() {});
                break;
            default:
                alert("imode error.");
        }
        cards = document.querySelectorAll('img');
        $('#next').off('click').on('click', press_next);
        $('#previous').off('click').on('click', press_previous);
    });
}

function search_sid() {
    var params = {
        sid: $("#sid").val(),
        dataset: dataset
    }; 
    var query = $.param(params);
    $.getJSON(`/search_sid?${query}`, function(result) {
        $("#sid-rst").text(result.rst);
    });
}

function press_next() {
   
    if (current_id < job_end) {
        increase_current_id();
    } else {
        alert("已是最大的id.");
    }
    $("#next").blur();

}

function press_previous() {
    
    if (current_id > job_start) {
        decrease_current_id();
    } else {
        alert("已是最小的id.");
    }
    $("#previous").blur();
    
}

function isIn(el) {
    var bound = el.getBoundingClientRect();
    var clientHeight = window.innerHeight;
    return bound.top <= clientHeight;
}

function check() {
    Array.from(cards).forEach(function(el){
        if(isIn(el)) {
            loadCard(el);
        }
    })
}

function loadCard(im) {
    if(!im.src){
        var source = im.dataset.src;
        im.src = source;
    }
}

// window.onload = 
window.onscroll = function () { 
    check();
}

function show_face_only() {
    $.getJSON(`/update_imode/${dataset}/${"face"}`, function( result ) {
        if (result.status !== "success") {
            alert(result.status);
        }
    });
    $("#body-btn").each(function() {
        $( this ).removeClass( "btn-primary" );
        $( this ).addClass( "btn-secondary" );
    });
    $("#hybrid-btn").each(function() {
        $( this ).removeClass( "btn-primary" );
        $( this ).addClass( "btn-secondary" );
    });
    $("#face-btn").each(function() {
        $( this ).removeClass( "btn-secondary" );
        $( this ).addClass( "btn-primary" );
    });
    $( ".fface" ).each(function() {
        $( this ).removeClass( "sr-only" );
      });
    $( ".fbody" ).each(function() {
        $( this ).addClass( "sr-only" );
      });
}

function show_body_only() {
    $.getJSON(`/update_imode/${dataset}/${"body"}`, function( result ) {
        if (result.status !== "success") {
            alert(result.status);
        }
    });
    $("#face-btn").each(function() {
        $( this ).removeClass( "btn-primary" );
        $( this ).addClass( "btn-secondary" );
    });
    $("#hybrid-btn").each(function() {
        $( this ).removeClass( "btn-primary" );
        $( this ).addClass( "btn-secondary" );
    });
    $("#body-btn").each(function() {
        $( this ).removeClass( "btn-secondary" );
        $( this ).addClass( "btn-primary" );
    });
    $( ".fbody" ).each(function() {
        $( this ).removeClass( "sr-only" );
      });
    $( ".fface" ).each(function() {
        $( this ).addClass( "sr-only" );
      });
}

function show_hybrid(func) {
    $("#face-btn").each(function() {
        $( this ).removeClass( "btn-primary" );
        $( this ).addClass( "btn-secondary" );
    });
    $("#body-btn").each(function() {
        $( this ).removeClass( "btn-primary" );
        $( this ).addClass( "btn-secondary" );
    });
    $("#hybrid-btn").each(function() {
        $( this ).removeClass( "btn-secondary" );
        $( this ).addClass( "btn-primary" );
    });
    $.getJSON(`/update_imode/${dataset}/${"hybrid"}`, function( result ) {
        if (result.status !== "success") {
            alert(result.status);
        }
    });
    func();
}

function remove_hide() {
    $( ".fbody" ).each(function() {
        $( this ).removeClass( "sr-only" );
    });
    $( ".fface" ).each(function() {
        $( this ).removeClass( "sr-only" );
    });
}

$(document).ready(function() {
    $.ajaxSetup({ cache: false });
    dataset = $("#dataset").html();
    imode = $("#imode").html();
    server_address = window.location.origin;
    name = $("#name").html();
    is_body = "on";
    img_type = $("#itype").html();
    current_id = Number($("#current").html());
    job_start = 0;
    job_end = Number($("#max-label").html());
    total_job = job_end - job_start;
    lbs = localStorage["lbs"];
    var request;
    if (lbs !== "") {
        lbs = lbs.split(",");
        current_id = Number(localStorage['current']);
        if (current_id > lbs.length) {
            alert("选定的label已显示完.");
        } else {
            current_rst = lbs[current_id - 1];
            request = `/templates-body/all/${dataset}/${current_rst}`;
            refresh_view(function () {
                reset_actions();
                center_view(request);
            });
        }
    } else {
        get_current_rst(function () {
            request = `/templates-body/all/${dataset}/${current_rst}`;
            refresh_view(function () {
                reset_actions();
                center_view(request);
            });
        }); 
    }
       
    $("#jumpSelect").change(function () {
        $( "#jumpSelect option:selected" ).each(function() {
            $("#goid").off("keypress");
            $("#jump button").off("click");
            if ($(this).val() === "label") { 
                $("#goid").keypress(enter_go_label);
                $("#goid").attr("placeholder", "label");
                $("#jump button").click(go_to_label);
            } else {
                $("#goid").keypress(enter_goid);
                $("#goid").attr("placeholder", "rst");
                $("#jump button").click(go_to_id);
            }
        });
    }).trigger( "change" );
});

$(document).keyup(function (event) {
    switch(event.keyCode) {
        case 37: //left
            $('#previous').trigger( "click" );
            return;
        case 39: //right
            $('#next').trigger( "click" );
            return;
    }
});
