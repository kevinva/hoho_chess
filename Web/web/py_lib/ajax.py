
def send(data, callback):
	ajax = javascript.XMLHttpRequest.new()
	def onerror(e):
		statusText = ajax.statusText.data()
		if statusText=='':
			statusText = '未能从服务器获取信息'
		callback({'error': statusText})
	def onreadystatechange(e):
		if (ajax.readyState.data() == 4): # XMLHttpRequest.DONE
			status = ajax.status.data()
			if ((status>=200) and (status<300)) or (status==304):
				callback(ajax.response.data())
			else:
				onerror(None)
	ajax.onreadystatechange = onreadystatechange
	ajax.onerror = onerror
	ajax.open('POST', '/ajax', True)
	ajax.setRequestHeader("Content-type", "application/x-www-form-urlencoded")
	ajax.responseType = 'json'
	post_data = {'data':javascript.JSON.stringify(data)}
	ajax.send(javascript.URLSearchParams.new(post_data))
