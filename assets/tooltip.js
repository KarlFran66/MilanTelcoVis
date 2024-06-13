window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.formatDate = function(value) {
    var date = new Date(value);
    return date.toISOString().split('T')[0]; // Format as YYYY-MM-DD
};

window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.formatTime = function(value) {
    var hours = Math.floor(value / 6);
    var minutes = (value % 6) * 10;
    return hours.toString().padStart(2, '0') + ':' + minutes.toString().padStart(2, '0');
}

