/**
 * trs
 * Authors: Michael Luo
 * Date: 07/07/2017
 * Desc:
 * Copyright (c) 2016 caiyunapp.com. All rights reserved.
 */
// pNodes = [];
(function () {
  // isScriptLoaded();
  // if (pNodes.length > 0)
  //   return
  // var trs_type = 0;

  var CLASSNAME = 'caiyun-trs';
  var pNodes = [];
  var hNodes = [];
  var spanNodes = [];
  var textNodes = [];
  var viewport = getViewport();
  var body = document.body;
  var time = Date.now();
  console.log(getPagearea(), viewport);
  // window.onscroll = function (e) {
  //   if (Date.now() - time > 5000) {
  //     console.log("æ£€æµ‹åˆ°é¡µé¢æ»šåŠ¨äº‹ä»¶:" + window.pageXOffset + " " + window.pageYOffset);
  //     searchText(body);
  //     cycle();
  //   }
  //   time = Date.now();
  // }

  // if (document.querySelector("." + CLASSNAME)) return

  searchText(body);
  cycle();

  //è§£æžHtmlï¼Œæå–æ–‡æœ¬ï¼Œç”Ÿæˆå¯¹åº”æ•°ç»„
  function searchText(element) {
    var nodeList = element.childNodes;
    // console.log("element.childNodes: ",  element.childNodes.length)

    // if (!isElementInViewport(element)) {
    //   console.log("Element not in Viewport return")
    //   return
    // }

    if (element.classList && element.classList.contains(CLASSNAME)) {
      console.log(element, "contains('caiyun-trs') å·²ç¿»è¯‘ return")
      return
    }

    // å¾ªçŽ¯éåŽ†å­å…ƒç´ èŠ‚ç‚¹
    for (var i = 0, l = nodeList.length; i < l; i++) {
      // if (nodeList[i].nodeName == "SPAN" || nodeList[i].nodeName == "LI" || nodeList[i].nodeName == "A") {
      //   spanNodes.push(nodeList[i]);
      //   continue
      // }
      // console.log(getElementViewTop(nodeList[i]))

      // if (getElementViewTop(nodeList[i]))
      if (!nodeList[i]) return;
      if (nodeList[i].classList && nodeList[i].classList.contains(CLASSNAME)) {
        console.log(nodeList[i], "contains('caiyun-trs') å·²ç¿»è¯‘ return")
        return
      }

      // æå–Hæ ‡ç­¾
      if (nodeList[i].nodeName == "H1" || nodeList[i].nodeName == "H2" || nodeList[i].nodeName == "H3"
        || nodeList[i].nodeName == "H4" || nodeList[i].nodeName == "H5"
        || nodeList[i].nodeName == "H6" && nodeList[i].innerText.trim() != "") {

        // Hæ ‡ç­¾ä¸­åŒ…å«SPANæ ‡ç­¾ç•¥è¿‡
        if (nodeList[i].lastChild && nodeList[i].lastChild.nodeName == "SPAN") {

        } else {
          hNodes.push(nodeList[i]);
          // element.removeChild(nodeList[i]);
          continue
        }
      }

      // æå–Pæ ‡ç­¾
      if (nodeList[i].nodeName == "P" && nodeList[i].innerText.trim() != "") {
        pNodes.push(nodeList[i]);
        continue;
      }

      // å…¶ä»–å½’äºŽæ–‡æœ¬ç±»åž‹æ•°ç»„
      if (nodeList[i].nodeType == Node.TEXT_NODE && nodeList[i].nodeValue.trim() != "") {
        // textArray.push(nodeList[i].nodeValue);
        textNodes.push(nodeList[i]);
        continue;
      } else if (nodeList[i].nodeType == Node.ELEMENT_NODE) {
        searchText(nodeList[i]);
        // if (isElementInViewport(nodeList[i])) {
        //   searchText(nodeList[i]);
        //   console.log("ElementInViewport", nodeList[i]);
        // } else {
        //   console.log("Element Not in Viewport. continue");
        //   continue
        // }
      }
    }

    // console.log(hNodes, pNodes, spanNodes, textNodes);
  }

  async function cycle() {
    console.log("cycle", hNodes, pNodes, textNodes);

    for (var i = 0, l = hNodes.length; i < l; i++) {
      translate(hNodes[i], 'h');
    }

    for (var i = 0, l = pNodes.length; i < l; i++) {
      await sleep(100);
      translate(pNodes[i], 'p');
    }

    // for (var i = 0, l = spanNodes.length; i < l; i++) {
    //   translate(spanNodes[i], 2);
    // }

    for(var i=0, l=textNodes.length; i<l; i++) {
      await sleep(100);
      translate(textNodes[i], 'text');
    }
  }

  //type=p, clone and insert target;
  //type=h, clone and insert target;
  //type=text, add target
  function translate(node, _type) {
    // if (!node.innerText)
    //   return
    var source = node.innerText;
    if (_type == 'text')
      source = node.nodeValue;

    // source = source.replace("[", "").replace("]", "");
    if (!source || source.trim().length < 2 || !isNaN(source)) {
      return
    }

    if (node.classList && node.classList.contains(CLASSNAME)) {
      console.log("translate return")
      return
    }

    var trans_type = "en2zh";
    if (isChinese(source)) {
      trans_type = "zh2en";
      // return
    }

    var xhr = createXMLHttpObject();
    xhr.open('POST', 'https://api.interpreter.caiyunai.com/v1/translator?model=onmt:v2', true);
    // xhr.setRequestHeader('X-Custom-Header', 'value');
    xhr.setRequestHeader('content-type', 'application/json');
    xhr.setRequestHeader('X-Authorization', 'token j1np9nb4h8jad0mi2odk');

    // æ³¨å†ŒæŽ¥å£å›žè°ƒå‡½æ•°
    xhr.onreadystatechange = function (event) {
      if (xhr.readyState == 4) {
        var res = xhr.status == 200 ? JSON.parse(xhr.responseText) : null;

        if (res && res.rc == 0) {
          var target = res.target;
          console.log(source, target);

          addClass(node, CLASSNAME);
          if (_type == 'p') {
            var clone = node.cloneNode(true);
            clone.innerText = target;
            node.parentNode.insertBefore(clone, node.nextSibling);
          } else if (_type == 'h') {
            target = trimDot(target);
            if (source == target) return;
            var clone = node.cloneNode(true);
            clone.innerText = target;
            node.parentNode.insertBefore(clone, node.nextSibling);
          } else {
            target = trimDot(target);
            if (source == target) return;
            // console.log("className", node.className);
            node.nodeValue = source + ' ' + target + ' ';
          }

        }
        else {
          console.error(res);
        }
      }
    };

    xhr.send(JSON.stringify({
      "source": cleanSource(source),
      "trans_type": trans_type,
      "request_id": document.URL || "web-translate"
      //"replaced": true,
      //"cached" : true
    }));

  }

  function isChinese(s) {
    var patrn = /[\u4E00-\u9FA5]|[\uFE30-\uFFA0]/gi;
    if (!patrn.exec(s)) {
      return false;
    }
    else {
      return true;
    }
  }

  function trimDot(str) {
    var char = str[str.length - 1];
    if (char == "." || char == "ã€‚" || char == "ï¼")
      str = str.substr(0, str.length - 1).trim();
    return str
  }

  function cleanSource(s) {
    return s.replace("[", "").replace("]", "").replace("(", "").replace(")", "");
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function addClass(obj, cls){
    var obj_class = obj.className || ''; // èŽ·å– class å†…å®¹.
    var blank = (obj_class != '') ? ' ' : '';//åˆ¤æ–­èŽ·å–åˆ°çš„ class æ˜¯å¦ä¸ºç©º, å¦‚æžœä¸ä¸ºç©ºåœ¨å‰é¢åŠ ä¸ª'ç©ºæ ¼'.
    var added = obj_class + blank + cls;//ç»„åˆåŽŸæ¥çš„ class å’Œéœ€è¦æ·»åŠ çš„ class.
    obj.className = added;//æ›¿æ¢åŽŸæ¥çš„ class.
  }

  function isElementInViewport(el) {
    var top = el.offsetTop;
    var left = el.offsetLeft;
    var width = el.offsetWidth;
    var height = el.offsetHeight;

    while(el.offsetParent) {
      el = el.offsetParent;
      top += el.offsetTop;
      left += el.offsetLeft;
    }

    console.log(el, top, left, top + height, left + width);

    return (
      top < (window.pageYOffset + window.innerHeight) &&
      left < (window.pageXOffset + window.innerWidth) &&
      (top + height) > window.pageYOffset &&
      (left + width) > window.pageXOffset
    );
  }

  function isElementInViewport2 (el) {
    var rect = el.getBoundingClientRect();
    console.log(el, rect);
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) && /*or $(window).height() */
      rect.right <= (window.innerWidth || document.documentElement.clientWidth) /*or $(window).width() */
    );
  }

  /**
   * åˆ›å»º XMLHttpRequest å¯¹è±¡ï¼Œå¤„ç†è·¨åŸŸè¯·æ±‚
   */
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

  function isScriptLoaded(url) {
    if (!url) url = ('https:' == document.location.protocol ? 'https://' : 'http://') + 'localhost:3666/dest/trs.js';
    var scripts = document.getElementsByTagName('script');
    console.log(scripts);
    for (var i = scripts.length; i--;) {
      if (scripts[i].src == url) {
        throw new Error("trs.js loaded")
        return true;
      }
    }
    return false;
  }

  var BACKFLAG = 'dataBack';

  // èŽ·å–æµè§ˆå™¨çª—å£å®½å’Œé«˜
  function getViewport(){
    if (document.compatMode == "BackCompat"){
      return {
        width: document.body.clientWidth,
        height: document.body.clientHeight
      }
    } else {
      return {
        width: document.documentElement.clientWidth,
        height: document.documentElement.clientHeight
      }
    }
  }

  // èŽ·å–ç½‘é¡µå®½å’Œé«˜
  function getPagearea(){
    if (document.compatMode == "BackCompat"){
      return {
        width: Math.max(document.body.scrollWidth,
          document.body.clientWidth),
        height: Math.max(document.body.scrollHeight,
          document.body.clientHeight)
      }
    } else {
      return {
        width: Math.max(document.documentElement.scrollWidth,
          document.documentElement.clientWidth),
        height: Math.max(document.documentElement.scrollHeight,
          document.documentElement.clientHeight)
      }
    }
  }

  // èŽ·å–å…ƒç´ ç›¸å¯¹é«˜åº¦
  function getElementViewTop(element){
    if (element.offsetTop == undefined || element.offsetParent == undefined)
      return

    var actualTop = element.offsetTop;
    var current = element.offsetParent;
    while (current !== null){
      actualTop += current. offsetTop;
      current = current.offsetParent;
    }
    if (document.compatMode == "BackCompat"){
      var elementScrollTop=document.body.scrollTop;
    } else {
      var elementScrollTop=document.documentElement.scrollTop;
    }
    return actualTop-elementScrollTop;
  }

  /**
   * äº‹ä»¶ç»‘å®š
   * @param object è¦ç»‘å®šäº‹ä»¶çš„å¯¹è±¡
   * @param eventName äº‹ä»¶åç§°
   * @param callback äº‹ä»¶å¤„ç†å‡½æ•°
   */
  var bind = function (object, eventName, callback) {
    if (!callback) {
      return;
    }
    if (object.addEventListener) {
      object.addEventListener(eventName, callback, false);
    } else if (object.attachEvent) {
      object.attachEvent('on' + eventName, callback);
    } else {
      object['on' + eventName] = callback;
    }
    return this;
  };
  /**
   * åˆ¤æ–­å¯¹è±¡æ˜¯å¦å‡½æ•°
   * @param obj å¾…æ£€æŸ¥å¯¹è±¡
   */
  var isFunction = function (obj) {
    return !!(Object.prototype.toString.call(obj) === "[object Function]");
  };

  /**
   * è·¨åŸŸé€šä¿¡å“åº”å¯¹è±¡
   */
  var Response = {
    /**
     * å“åº”è¯·æ±‚ä¿¡æ¯
     * @param callback
     */
    onMessage: function (callback) {
      if (!isFunction(callback)) {
        callback = function () {
        };
      }
      bind(window, 'message', function (eve) {
        callback(eve);
      });
    },
    /**
     * å‘å¦ä¸€ä¸ªåŸŸå‘é€è¯·æ±‚
     * @param responseData
     */
    sendMessage: function (responseData) {
      parent.postMessage(JSON.stringify(responseData), '*');
    }
  };
  /**
   * æœ¬åœ°å­˜å‚¨ã€‚ æ‰€æœ‰æœ¬åœ°å­˜å‚¨ç›¸å…³æ•°æ®å­˜å‚¨åˆ° youdao çš„åŸŸä¸‹ï¼Œè¿™æ ·æ‰èƒ½åšåˆ°ç”¨æˆ·è®¾ç½®ä¸ŽåŸŸæ— å…³ã€‚
   * @param key é”®
   * @param value å€¼
   */
  var storage = function (key, value) {
    /**
     * html5 ä¸­çš„æœ¬åœ°å­˜å‚¨æ–¹å¼
     * @param key
     * @param value
     */
    var html5LocalStorage = function (key, value) {
      var store = window.localStorage;
      if (value === undefined) {
        return store.getItem(key);
      }
      if (key !== undefined && value !== undefined) {
        store.setItem(key, value);
        return value;
      }
    };
    /**
     * IE æœ¬åœ°å­˜å‚¨æ–¹å¼ userData
     * @param key
     * @param value
     */
    var userdata = function (key, value) {
      var store = document.documentElement;
      store.addBehavior("#default#userData");
      if (value === undefined) {
        store.load("fanyiweb2");
        return store.getAttribute(key);
      }
      if (key !== undefined && value !== undefined) {
        store.setAttribute(key, value);
        store.save("fanyiweb2");
        return value;
      }
    };
    if (!!window.localStorage) {
      return html5LocalStorage(key, value);
    }
    if (!!document.documentElement.addBehavior) {
      return userdata(key, value);
    }
  };

  /**
   *
   */
  var handleMessage = function () {
    /**
     * å°† request ä¸­çš„ data è½¬ä¸ºå¯¹è±¡
     * @param request
     */
    var initData = function (request) {
      var dataArray = request.data, data = {};
      if (typeof dataArray === 'string') {
        dataArray = dataArray.split('&');
      }

      for (var i = 0; i < dataArray.length; i++) {
        var d = dataArray[i].split('=');
        data[d[0]] = d[1];
      }
      return data;
    };
    /**
     * æ‰€æœ‰è¯·æ±‚çš„å¤„ç†å‡½æ•°ï¼Œè¯·æ±‚çš„å¤„ç†å‡½æ•°ä¸Ž request.handler å±žæ€§å€¼åº”ä¿æŒä¸€è‡´
     */
    var handlers = {
      /**
       * èŽ·å–ç¿»è¯‘çš„æŸ¥è¯¢ç»“æžœ
       * @param request è¯·æ±‚æ•°æ®
       */
      translate: function (request) {
        var xhr = createXMLHttpObject();
        xhr.onreadystatechange = function (event) {
          if (xhr.readyState == 4) {
            var data = xhr.status == 200 ? xhr.responseText : null;
            Response.sendMessage({
              'handler': BACKFLAG,
              'response': data,
              'index': request.index
            });
          }
        };
        xhr.open(request.type, request.url, true);

        if (request.type === 'POST') {
          xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
          xhr.send(request.data);
        } else {
          xhr.send(null);
        }
      },
      /**
       * æœ¬åœ°å­˜å‚¨
       * @param request è¯·æ±‚ä¿¡æ¯
       */
      localStorage: function (request) {
        var data = initData(request);
        var result = decodeURIComponent(storage(data.key, data.value));
        Response.sendMessage({
          'handler': BACKFLAG,
          'response': result,
          'index': request.index
        });
      }
    };
    // return function(request) {
    //   if (!!!handlers[request.handler]) {
    // throw new Error('ç±»åˆ«ä¸º ' + request.handler + ' è·¨åŸŸè¯·æ±‚å¤„ç†å‡½æ•°ä¸å­˜åœ¨ï¼');
    // }
    // handlers[request.handler](request);
    // };
  }();

  /**
   * æ³¨å†Œæ¶ˆæ¯å¤„ç†æœºåˆ¶
   */
  // Response.onMessage(function(eve) {
  //   handleMessage(JSON.parse(eve.data));
  // });

  // Response.sendMessage({handler:'transferStationReady'});

})();
