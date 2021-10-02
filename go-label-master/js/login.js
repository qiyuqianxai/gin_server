function go_to_doc() {
    window.location.href = "https://gitlab-research.cloudwalk.work/Char/go-label/blob/master/README.md";
}

function submit_info() {
  let face = $("#selDataset").val();
  let body = $("#selBDataset").val();
  let roc = $("#selRoc").val();
  let lbs = $("#Labels").val().trim();
  localStorage.setItem('lbs', lbs);
  if (lbs !== "") {
    localStorage.setItem('current', 1);
  }
  if (!face && !body && !roc) {
    alert("请至少选择一个数据集.");
  } else if (face && body || face && roc || body && roc) {
    alert("只能选择一个数据集.");
  } else {
    $(this).prop("disabled", true);
    $(this).html(
      `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...`
    );
    $("form").submit();
  }
}

$("#start").click(submit_info);