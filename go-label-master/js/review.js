var cards;
var red = `rgba(255, 0, 0, .5)`;
var green = `rgba(0, 255, 0, .5)`;
var blue = `rgba(0, 0, 255, .5)`;

function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 15)];
    }
    return color;
}

function reset_actions() {
    $('#next').off('click', press_next);
    $('#previous').off('click', press_previous);

    clean_center = {};
}

function center_view(request, action = "next") {
    $.getJSON(request, function( result ) {
        var imgpath = result.imgpath;
        var meta = result.meta;
        var info = result.info;
        var i;
        var image_name;
        var image_src;
        var origin_image_src;
        var short_name;
        var inf;
        var color;
        var colors = [`rgba(255, 255, 255, .5)`, red, green, blue];
        $("#classid").text((current_id - job_start).toString() + ' / ' + total_job.toString());
        if (!isNaN(imgpath.length)) {
            $("#rst").text(current_rst + ": 共" + imgpath.length + "张");
            if (!isNaN(meta.length) && (meta.length === imgpath.length)) {  
                var max_meta = 0;
                for (i = 0; i < meta.length; i++) {
                    if (Number(meta[i]) > max_meta) {
                        max_meta = Number(meta[i]);
                    }
                }
                if (max_meta >= colors.length) {
                    var size = max_meta - colors.length + 1;
                    for (i = 0; i < size; i++) {
                        colors.push(getRandomColor());
                    }
                }
            }
            for (i = 0; i < imgpath.length; i++) {
                clean_center["face"+i.toString()] = i;
                image_name = imgpath[i];
                inf = null;
                color = null;
                if (!isNaN(meta.length) && (meta.length === imgpath.length)) {   
                    color = colors[Number(meta[i])]; 
                } 
                if (!isNaN(Object.keys(info).length) && (Object.keys(info).length === imgpath.length)) {
                    inf = info[i];
                } 
                image_src = gen_image_src(dataset, image_name, img_type, true);
                origin_image_src = gen_image_src(dataset, image_name, img_type, false);
                short_name = image_name.substring(image_name.lastIndexOf('/')+1);
                add_to_gallery(i, image_src, short_name, origin_image_src, color, inf);
            }
        }
        cards = document.querySelectorAll('img');
        $('#next').off('click').on('click', press_next);
        $('#previous').off('click').on('click', press_previous);
    });
}

function press_next() {
    
    if (current_id < job_end) {
        increase_current_id();
        // refresh_view(function () {
        //     reset_actions();
        //     center_view(gen_info(), "next");
        // });
    } else {
        //reset_actions();
        alert("已是最大的id.");
    }
    $("#next").blur();
    
}

function press_previous() {
   
    if (current_id > job_start) {
        decrease_current_id();
        // refresh_view(function () {
        //     reset_actions();
        //     center_view(gen_info(), "previous");
        // });
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
    // if (el.classList.contains('sr-only')) {
    //     el.classList.remove('sr-only');
    //     var im = el.getElementsByTagName('img')[0];
        if(!im.src){
            var source = im.dataset.src;
            im.src = source;
        }
    // } 
}

// window.onload = 
window.onscroll = function () { 
    check();
}

$(document).ready(function() {
    dataset = $("#dataset").html();
    server_address = window.location.origin;
    name = $("#name").html();
    img_type = $("#itype").html();
    current_id = Number($("#current").html());
    job_start = 0;
    job_end = Number($("#max-label").html());
    total_job = job_end - job_start;
    is_body = "off";
    lbs = localStorage["lbs"];
    var request;
    if (!lbs) {
        get_current_rst(function (){
            request = `/templates/all/${dataset}/${current_rst}`;
            refresh_view(function () {
                reset_actions();
                center_view(request);
            });
        }); 
    }
    else if (lbs !== null) {
        lbs = lbs.split(",");
        current_id = Number(localStorage['current']);
        if (current_id > lbs.length) {
            alert("选定的label已显示完.");
        } else {
            current_rst = lbs[current_id - 1];
            request = `/templates/all/${dataset}/${current_rst}`;
            refresh_view(function () {
                reset_actions();
                center_view(request);
            });
        }
    } 
       
    $("#jumpSelect").change(function () {
        $( "#jumpSelect option:selected" ).each(function() {
            $("#goid").off("keypress");
            $("#jump button").off("click");
            if ($(this).val() === "label") { 
                $("#goid").keypress(enter_go_label);
                $("#goid").attr("placeholder", "index");
                $("#jump button").click(go_to_label);
            } else {
                $("#goid").keypress(enter_goid);
                $("#goid").attr("placeholder", "label");
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
