// Nav Menu : Function click on a 'button' and set 'button' active, open or close sub-menus as required.
// to debug - use alert( "Text message." );
// 1st level menu click

( function( $ ) {
	"use strict";
$( document ).ready(function() {
$('#cssmenu > ul > li > a').click(function() {
  $('#cssmenu li').removeClass('active');
  $(this).closest('li').addClass('active');	
  var checkElement = $(this).next();
  if((checkElement.is('ul')) && (checkElement.is(':visible'))) {
	$(this).closest('li').removeClass('active');
    checkElement.slideUp('normal');
  }
  if((checkElement.is('ul')) && (!checkElement.is(':visible'))) {
	$('#cssmenu ul ul:visible').slideUp('normal');
	$('#cssmenu ul ul ul:visible').slideUp('normal');
	$('#cssmenu ul ul ul ul:visible').slideUp('normal');
	checkElement.slideDown('normal');
  }
  if($(this).closest('li').find('ul').children().length === 0) {
	return true;
  } else {
	return false;	
  }		
});
});
} )( jQuery );

// 2nd level menu click
( function( $ ) {
	"use strict";
$( document ).ready(function() {
$('#cssmenu > ul > li > ul > li > a').click(function() {
  $('#cssmenu li').removeClass('active');
  $(this).closest('li').addClass('active');	
  var checkElement = $(this).next();
  if((checkElement.is('ul')) && (checkElement.is(':visible'))) {
	$(this).closest('li').removeClass('active');
    checkElement.slideUp('normal');
  }
  if((checkElement.is('ul')) && (!checkElement.is(':visible'))) {
	$('#cssmenu ul ul ul:visible').slideUp('normal');
	$('#cssmenu ul ul ul ul:visible').slideUp('normal');
    checkElement.slideDown('normal');
  }
  if($(this).closest('li').find('ul').children().length === 0) {
	return true;
  } else {
	return false;	
  }		
});
});
} )( jQuery );

// 3rd level menu click 
( function( $ ) {
	"use strict";
$( document ).ready(function() {
$('#cssmenu > ul > li > ul > li > ul > li > a').click(function() {
  $('#cssmenu li').removeClass('active');
  $(this).closest('li').addClass('active');	
  var checkElement = $(this).next();
  if((checkElement.is('ul')) && (checkElement.is(':visible'))) {
	$(this).closest('li').removeClass('active');
    checkElement.slideUp('normal');
  }
  if((checkElement.is('ul')) && (!checkElement.is(':visible'))) {
	$('#cssmenu ul ul ul ul:visible').slideUp('normal');
    checkElement.slideDown('normal');
  }
  if($(this).closest('li').find('ul').children().length === 0) {
	return true;
  } else {
	return false;	
  }		
});
});
} )( jQuery );