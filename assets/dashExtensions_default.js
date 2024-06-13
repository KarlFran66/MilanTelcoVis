window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function(feature, context) {
            const {
                classes,
                colorscale,
                style,
                colorProp
            } = context.hideout;
            const value = feature.properties[colorProp];
            for (let i = classes.length - 1; i >= 0; --i) {
                if (value >= classes[i]) {
                    style.fillColor = colorscale[i];
                    break;
                }
            }
            return style;
        }
    }
});