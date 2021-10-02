function submit_info() {
    let db = $("#selDB").val();
    if (!db) {
        alert("请选择一个数据集.");
    } else {
        $(this).prop("disabled", true);
        $(this).html(
            `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...`
        );
        $("form").submit();
    }
}

$("#start").click(submit_info);