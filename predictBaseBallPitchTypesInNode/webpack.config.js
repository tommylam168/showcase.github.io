const path=require('path');

module.exports = {
	entry:'./client.js',
	output:{path:path.resolve(__dirname,'dist'),filename:'bundle.js'},
	devServer:{historyApiFallback:true,host:'0.0.0.0'}
	}