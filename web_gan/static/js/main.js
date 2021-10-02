// current src image
current_image = ""

$(function () {
    // 加载按键信息
    set_click_response();
    get_base_info();
});


// load src_imgs list
function get_base_info() {
    $.ajax({
        url:"/get_base_info/",
        contentType: "application/json; charset=utf-8",
        type:"GET",
        cache:false,
        success:function(data){
            //每次加载时重置一些参数

            var img_list = data['img_list'];//视频列表
            console.log("imgs",img_list)
            $('#current-image').empty();

            $.each(img_list, function (i, img_name) {
                $("#current-image").append("<option value=" + img_name + ">"+img_name+"</option>");
            });
            $('#current-image').on('change', function(e){
                if (e.originalEvent) {
                    let selected_img = $(this).find("option:selected").val();
                    if(selected_img !== current_image)
                    {
                        var img = new Image()
                        // 改变图片的src
                        img.src = "src_imgs/"+selected_img
                        // 加载完成执行
                        img.onload = function(){
                            $('#src_img').attr("src", "src_imgs/"+selected_img);
                            var windowW = $(window).width()*0.45;//获取当前窗口宽度
                            var windowH = $(window).height()*0.6;//获取当前窗口高度
                            var realWidth = img.width;//获取图片真实宽度
                            var realHeight = img.height;//获取图片真实高度
                            var scale = Math.max(realWidth/windowW,realHeight/windowH);//缩放尺寸，当图片真实宽度和高度大于窗口宽度和高度时进行缩放
                            console.log(realWidth,realHeight,windowW,windowH,scale)
                            $('#src_img').css({"width":realWidth/scale,"height":realHeight/scale});
                            current_image = selected_img;
                        }

                    }
                    // console.log(current_video);
                }
            });

        },
        error:function(data){
            alert("数据加载出错，请联系管理员！");
            top.location.reload();
        }
    });
}

// 设置各个功能响应
function set_click_response() {
    // ai功能响应
    $('#convert').blur().on("click",function () {
        convert_img();
    });

    $('#upload_image').blur().on("click",function () {
        upload_image();
    })
}

function convert_img(){
    var post_data = JSON.stringify({
        src_img: current_image,
        gan_eyes_color: $('#gan-eyes_color').find("option:selected").val(),
        gan_gender: $('#gan-gender').find("option:selected").val(),
        gan_hair_color: $('#gan-hair_color').find("option:selected").val(),
    });
    $.ajax({
        url: "/convert_img/",
        type: "POST",
        cache:false,
        data:post_data,
        success: function (data) {
            post_data = JSON.parse(post_data);
            console.log(post_data)
            var dest_img = "dest_imgs/"+current_image.replace(".jpg","").replace(".jpeg","").replace("png","")
                +"_"+ post_data.gan_gender+"_"+post_data.gan_eyes_color+"_"+post_data.gan_hair_color+".jpg"
            var img = new Image()
            // 改变图片的src
            img.src = dest_img
            // 加载完成执行
            img.onload = function(){
                $('#dest_img').attr("src", dest_img);
                var windowW = $(window).width()*0.45;//获取当前窗口宽度
                var windowH = $(window).height()*0.6;//获取当前窗口高度
                var realWidth = img.width;//获取图片真实宽度
                var realHeight = img.height;//获取图片真实高度
                var scale = Math.max(realWidth/windowW,realHeight/windowH);//缩放尺寸，当图片真实宽度和高度大于窗口宽度和高度时进行缩放
                console.log(realWidth,realHeight,windowW,windowH,scale)
                $('#dest_img').css({"width":realWidth/scale,"height":realHeight/scale});
            }
            alert("generate success!")
        },
        error: function (data) {
            alert("出现错误，请联系管理员！");
        }
    })
}


// 上传视频到服务器
function upload_image() {
    //首先监听input框的变动，选中一个新的文件会触发change事件
    document.querySelector("#file").addEventListener("change",function () {
        //获取到选中的文件
        var file = document.querySelector("#file").files[0];
        var name = file.name;

        //创建formdata对象
        var formdata = new FormData();
        formdata.append("file",file);
        //创建xhr，使用ajax进行文件上传
        var xhr = new XMLHttpRequest();
        xhr.open("post","/upload_file/");
        //回调
        xhr.onreadystatechange = function () {
            if (xhr.readyState==4 && xhr.status==200){
                alert("上传成功！");
                get_base_info();
            }
        }
        //获取上传的进度
        xhr.upload.onprogress = function (event) {
            if(event.lengthComputable){
                var percent = event.loaded/event.total *100;
                document.querySelector("#up_progress .progress-item").style.width = percent+"%";
            }
        }
        //将formdata上传
        xhr.send(formdata);
    });
}
