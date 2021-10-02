$(document).ready(function() {
    var meta = decodeURI($("#meta").html()).split(",");
    var rect = $("#rect").html();
    var sid = $("#sid").html();
    var image = document.getElementById('source');
    var camera = $("#camera").html();
    var meta_table = `<table border=\"2\" cellpadding=\"12\" cellspacing=\"5\"><tbody> \
                    <tr><th>sid</th><td>${sid}</td></tr></tbody></table> \
                    <table border=\"2\" cellpadding=\"10\" cellspacing=\"5\"><tbody> \
                    <th>itype</th><th>camera</th><th>camera_id</th><th>track_id</th> \
                    <th>timestamp</th><th>ts_beg</th><th>ts_end</th> \
                    <th>score</th></tr> \
                    <tr><td>${meta[0]}</td><td>${camera}</td><td>${meta[1]}</td><td>${meta[2]}</td> \
                    <td>${meta[3]}</td><td>${meta[4]}</td><td>${meta[5]}</td> \
                    <td>${meta[6]}</td></tr> \
                    </tbody></table>`;
    $("#card-meta").append(meta_table);
    image.onload = function() {
        var canvas = document.getElementById("canvas");
        canvas.width = image.width*0.5;
        canvas.height = image.height*0.5;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(image, 0, 0, image.width*0.5, image.height*0.5);
        if (rect !== "-1") {
            var box = rect.split(",");
            ctx.beginPath();
            ctx.lineWidth = "2";
            ctx.strokeStyle = "red";
            ctx.rect(Number(box[0])*0.5, Number(box[1])*0.5, Number(box[2])*0.5, Number(box[3])*0.5);
            ctx.stroke();
        }
    }
    image.src = image.dataset.src;
});