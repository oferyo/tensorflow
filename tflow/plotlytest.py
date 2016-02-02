import plotly.tools as tls
import plotly.plotly as py



in_sample_size = 200

x_data = np.random.randn(in_sample_size).astype("float32")
y_data = np.zeros(in_sample_size)

y_data[1:in_sample_size] = x_data[0:in_sample_size-1]
y_data[0] = 0

y_data = np.matrix(y_data).reshape((in_sample_size, 1))



tls.set_credentials_file(
        username="ofer.goldfish",
        api_key="3k0rl3daea")


trace = dict(x=x_data, y=y_data)
data = [trace]
py.plot(data, filename='ia_county_populations')
py.plot(data)
