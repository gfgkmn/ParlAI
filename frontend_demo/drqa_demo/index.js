(function() {

    var PNUM = 21 
    var paragraph_id = 0;
    var titles;
    var contextss;
    var context_questions;

    window.onload = function() {
        displayQuestion();
        load();
    }

    function clearField() {
        var qadiv = document.getElementById("qa");
        qadiv.innerHTML = "";
        displayQuestion();
    }

    function createXMLHttpObject() {
        var XHRFactory = [
            function () {
                return new XMLHttpRequest();
            },
            function () {
                return new ActiveXObject('Msxml2.XMLHTTP');
            },
            function () {
                return new ActiveXObject('Msxml3.XMLHTTP');
            },
            function () {
                return new ActiveXObject('Microsoft.XMLHTTP');
            }
        ];
        var xhr = false;
        for (var i = 0; i < XHRFactory.length; i++) {
            try {
                xhr = XHRFactory[i]();
            } catch (e) {
                continue;
            }
            break;
        }
        return xhr;
    }

    function load() {
        sendAjax("/select", {}, handleParagraph);
    }

    function loadDropdown() {
        var dropdown = document.getElementById("selectArticle");
        for (var i = 0; i < PNUM; i++) {
            var opt = document.createElement("option");
            opt.value = parseInt(i);
            opt.innerHTML = titles[i];
            dropdown.appendChild(opt);
        }
        paragraph_id = 0;
        _loadParagraph();
        dropdown.onchange = loadParagraph;
    }

    function loadButtons() {
        var field = document.getElementById("button-field");
        for (var i = 0; i < 10; i++) {
            var b = document.createElement("button");
            b.type = "button";
            b.classList.add("btn");
            b.classList.add("btn-sm");
            if (i % 2 == 0) b.classList.add("btn-primary");
            else b.classList.add("btn-default");
            b.id = parseInt(i);
            b.innerHTML = "Para" + parseInt(i);
            b.onclick = loadParagraph;
            field.appendChild(b);
        }
    }

    function displayQuestion() {
        var form = document.createElement("div");
        form.id = "current";
        // label for Question
        var label = document.createElement("h4");
        var span = document.createElement("span");
        span.classList.add("label");
        span.classList.add("label-primary");
        span.innerHTML = "Question";
        // input for Qustion
        var input = document.createElement("input");
        input.type = "text";
        input.name = "question";
        input.classList.add("form-control");
        input.id = "question";
        // button to submit
        var button = document.createElement("button");
        button.type = "button";
        button.classList.add("btn");
        button.classList.add("btn-sm");
        button.classList.add("btn-default");
        button.innerHTML = "submit";
        button.id = "submit";
        // button to clear
        var clear = document.createElement("button");
        clear.type = "button";
        clear.classList.add("btn");
        clear.classList.add("btn-sm");
        clear.classList.add("btn-default");
        clear.innerHTML = "clear";
        clear.id = "clear";
        // button to translate 
        var translate = document.createElement("button");
        translate.type = "button";
        translate.classList.add("btn");
        translate.classList.add("btn-sm");
        translate.classList.add("btn-default");
        translate.innerHTML = "translate";
        translate.id = "translate";
        // loading
        var loading = document.createElement("div");
        loading.id = "loading";
        loading.style.display = "none";
        var img = document.createElement("img");
        img.src = "loading.gif";
        img.alt = "icon";
        loading.appendChild(img);
        loading.innerHTML = loading.innerHTML + "loading";
        // appendChild
        form.appendChild(label);
        label.appendChild(span)
        form.appendChild(input);
        form.appendChild(button);
        form.appendChild(clear);
        form.appendChild(translate);
        form.appendChild(loading);
        var qadiv = document.getElementById("qa");
        qadiv.append(form);

        button.onclick = loadAnswer;
        clear.onclick = clearField;
        translate.onclick = translateTriple;
    }

    function displayAnswer(answer) {
        var div = document.createElement("div");
        var label = document.createElement("h4");
        var span = document.createElement("span");
        span.classList.add("label");
        span.classList.add("label-primary");
        span.innerHTML = "Answer";
        var input = document.createElement("textarea");
        input.style = "resize:none";
        input.readOnly = true;
        input.classList.add("form-control");
        input.name = "answer";
        input.innerHTML = answer;
        div.appendChild(label);
        label.appendChild(span);
        div.appendChild(input);

        var qadiv = document.getElementById("qa");
        qadiv.append(div);
    }

    function loadAnswer() {
        document.getElementById("loading").style.display = "block";
        var data = {
            paragraph: $("#paragraph").val(),
            question: $("#question").val()
        };
        sendAjax("/submit", data, handleAnswer);
    }


    function translateTriple() {
        translateContext($("#paragraph").val(), 'paragraph_t');
        translateContext($("input[name='question']")[0].value, 'question_t');
        translateContext($("textarea[name='answer']")[0].value, 'answer_t');

        var img_t = document.getElementById("img_t");
        img_t.src = "attention_vis/vis.png?" + new Date().getTime();
    }

    function translateContext(source_text, target_id) {
        // var source_text = $("#paragraph").val();
        // var target_id = 'paragraph_t';

        var xhr = createXMLHttpObject();
        xhr.open('POST', 'https://api.interpreter.caiyunai.com/v1/translator?model=onmt:v2', true);
        xhr.setRequestHeader('content-type', 'application/json');
        xhr.setRequestHeader('X-Authorization', 'token j1np9nb4h8jad0mi2odk');


        xhr.onreadystatechange = function (event) {
            if (xhr.readyState == 4) {
                var res = xhr.status == 200 ? JSON.parse(xhr.responseText) : null;

                if (res && res.rc == 0) {
                    var target = res.target;
                    var curr = document.getElementById(target_id);
                    curr.innerHTML = target;
                    console.log(target);
                }
                else {
                    console.error(res);
                }
            }
        };

        xhr.send(JSON.stringify({
            "source": source_text,
            // question: $("input[name='question']")[0].val(),
            // answer: $("input[name='answer']")[0].val(),
            "trans_type": "en2zh",
            "request_id": "a11111",
            "replaced": true,
            "cached": true
        }));
    }


    function handleAnswer(answer) {
        var curr = document.getElementById("current");
        curr.removeChild(document.getElementById("submit"));
        curr.removeChild(document.getElementById("clear"));
        curr.removeChild(document.getElementById("translate"));
        curr.removeChild(document.getElementById("loading"));
        displayAnswer(answer);
        var q = document.getElementById("question");
        q.id = "";
        q.readOnly = true;

        var clear = document.createElement("button");
        clear.type = "button";
        clear.classList.add("btn");
        clear.classList.add("btn-sm");
        clear.classList.add("btn-default");
        clear.innerHTML = "new question!";
        clear.id = "clear";
        var translate = document.createElement("button");
        translate.type = "button";
        translate.classList.add("btn");
        translate.classList.add("btn-sm");
        translate.classList.add("btn-default");
        translate.innerHTML = "translate";
        translate.id = "translate";
        curr.appendChild(clear);
        curr.appendChild(translate);
        clear.onclick = clearField;
        translate.onclick = translateTriple;
        translateTriple()

    }

    function loadParagraph() {
        paragraph_id = this.value;
        _loadParagraph();
    }

    function _loadParagraph() {
        clearField();
        if (paragraph_id == 0) p_id = 1;
        document.getElementById("paragraph").value = contextss[paragraph_id];
        document.getElementById("question").value = context_questions[paragraph_id];
    }

    function handleParagraph(data) {
	console.log('------------------');
	console.log(data)
        titles = data.titles;
        contextss = data.contextss;
        context_questions = data.context_questions;
        loadDropdown();
    }

    function sendAjax(url, data, handle) {
	var trueurl = 'http://127.0.0.1:3377' + url;
        $.getJSON(trueurl, data, function(response) {
            handle(response.result);
        });
    }

})();
